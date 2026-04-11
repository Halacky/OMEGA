# import logging
# from typing import Optional, Dict

# import numpy as np
# from scipy import signal
# from scipy.fft import rfft, rfftfreq

# try:
#     import torch
# except ImportError:
#     torch = None


# class PowerfulEMGFeatures:
#     """
#     Оптимизированный извлекатель "мощных" EMG‑признаков.

#     Ключевые оптимизации:
#     - Векторные операции по всем каналам.
#     - Предвычисленные полосовые фильтры (sos) и фильтрация всех каналов разом.
#     - FFT по всем каналам одним вызовом.
#     - Опциональное использование GPU (torch) для TD/FD‑блоков.
#     - Энтропии считаются только на даунсэмплированном сигнале (по желанию).
#     """

#     def __init__(
#         self,
#         sampling_rate: int = 2000,
#         logger: Optional[logging.Logger] = None,
#         use_torch: bool = False,
#         device: Optional[str] = None,
#         use_entropy: bool = True,
#         entropy_downsample: int = 4,
#     ):
#         """
#         Args:
#             sampling_rate: частота дискретизации EMG.
#             use_torch: пытаться использовать torch для ускорения (если доступен).
#             device: 'cuda', 'cpu' или None (авто).
#             use_entropy: считать ли sample/approx entropy (медленные признаки).
#             entropy_downsample: во сколько раз даунсэмплировать сигнал перед энтропиями.
#         """
#         self.fs = sampling_rate
#         self.logger = logger

#         # Torch / GPU настройки
#         self.use_torch = bool(use_torch and (torch is not None))
#         if self.use_torch:
#             if device is None:
#                 self.device = "cuda" if torch.cuda.is_available() else "cpu"
#             else:
#                 self.device = device
#         else:
#             self.device = "cpu"

#         if self.logger:
#             self.logger.info(
#                 f"[PowerfulEMGFeatures] use_torch={self.use_torch}, device={self.device}"
#             )

#         # Time-frequency фильтры (предвычисляем SOS)
#         nyquist = 0.5 * self.fs
#         self.tf_bands = [
#             (20, 50),      # Low
#             (50, 100),     # Mid-Low
#             (100, 200),    # Mid-High
#             (200, 450),    # High
#         ]
#         self.tf_sos_filters = []
#         for low, high in self.tf_bands:
#             high_ = min(high, 0.45 * self.fs)
#             wp = [low / nyquist, high_ / nyquist]
#             sos = signal.butter(4, wp, btype="band", output="sos")
#             self.tf_sos_filters.append(sos)

#         # Настройки энтропии
#         self.use_entropy = use_entropy
#         self.entropy_downsample = max(1, int(entropy_downsample))

#         # Кэш для частотных сеток и масок: ключ = T
#         self._freq_cache: Dict[int, Dict[str, np.ndarray]] = {}

#     # ------------------------------------------------------------------ #
#     #                          Публичный метод                           #
#     # ------------------------------------------------------------------ #

#     def extract_batch(self, X: np.ndarray, batch_size: int = None) -> np.ndarray:
#         """
#         Батчевая версия извлечения фич.
#         X: (N, T, C)
#         Возвращает: (N, F)
#         По умолчанию делается полностью в NumPy (CPU), но можно вызывать и при use_torch=True.
#         """
#         if X.ndim != 3:
#             raise ValueError(f"Expected X shape (N, T, C), got {X.shape}")
#         N, T, C = X.shape

#         # Если batch_size указан — обрабатываем кусками (для экономии памяти),
#         # иначе — сразу весь массив.
#         if batch_size is None or batch_size >= N:
#             batch_indices = [(0, N)]
#         else:
#             batch_indices = []
#             for i in range(0, N, batch_size):
#                 batch_indices.append((i, min(N, i + batch_size)))

#         feats_all = []
#         for start, end in batch_indices:
#             Xb = X[start:end]  # (Nb, T, C)
#             Nb = Xb.shape[0]

#             # ------------------- TD (многомасштабные time-domain) -------------------
#             # Реализуем векторизованно, повторяя логику _extract_multiscale_td
#             third = T // 3
#             parts = [
#                 Xb[:, :third, :],
#                 Xb[:, third:2 * third, :],
#                 Xb[:, 2 * third :, :],
#             ]
#             td_feats_parts = []
#             for part in parts:
#                 L = part.shape[1]
#                 if L == 0:
#                     continue
#                 mav = np.mean(np.abs(part), axis=1)          # (Nb, C)
#                 rms = np.sqrt(np.mean(part ** 2, axis=1))    # (Nb, C)
#                 std = np.std(part, axis=1)                   # (Nb, C)
#                 if L > 1:
#                     diff = np.diff(part, axis=1)             # (Nb, L-1, C)
#                     wl = np.sum(np.abs(diff), axis=1)        # (Nb, C)
#                     signs = np.sign(part)
#                     zc = (np.diff(signs, axis=1) != 0).sum(axis=1) / (L + 1e-12)
#                 else:
#                     wl = np.zeros((Nb, C), dtype=float)
#                     zc = np.zeros((Nb, C), dtype=float)
#                 feats_part = np.stack([mav, rms, std, wl, zc], axis=2)  # (Nb, C, 5)
#                 td_feats_parts.append(feats_part)
#             if td_feats_parts:
#                 td_feats = np.concatenate(td_feats_parts, axis=2)  # (Nb, C, 5*num_parts)
#                 td_feats_flat = td_feats.reshape(Nb, -1)           # (Nb, F_td)
#             else:
#                 td_feats_flat = np.zeros((Nb, 0), dtype=float)

#             # ------------------- FD (частотные признаки) -------------------
#             # Векторизуем _extract_frequency: FFT по батчу
#             fft_vals = np.fft.rfft(Xb, axis=1)                  # (Nb, F, C)
#             psd = np.abs(fft_vals) ** 2                         # (Nb, F, C)
#             psd_sum = psd.sum(axis=1, keepdims=True) + 1e-12    # (Nb, 1, C)
#             psd_norm = psd / psd_sum                            # (Nb, F, C)

#             freqs = np.fft.rfftfreq(T, 1 / self.fs)             # (F,)
#             mean_freq = (psd_norm * freqs[None, :, None]).sum(axis=1)  # (Nb, C)

#             cumsum_psd = np.cumsum(psd_norm, axis=1)            # (Nb, F, C)
#             median_idx = (cumsum_psd >= 0.5).argmax(axis=1)     # (Nb, C)
#             median_freq = freqs[median_idx]                     # (Nb, C)

#             peak_idx = psd.argmax(axis=1)                       # (Nb, C)
#             peak_freq = freqs[peak_idx]                         # (Nb, C)

#             spectral_entropy = -(psd_norm * np.log(psd_norm + 1e-12)).sum(axis=1)  # (Nb, C)

#             freq_centered = freqs[None, :, None] - mean_freq[:, None, :]          # (Nb, F, C)
#             spectral_variance = (psd_norm * (freq_centered ** 2)).sum(axis=1)     # (Nb, C)
#             spectral_skewness = (psd_norm * (freq_centered ** 3)).sum(axis=1) / (
#                 spectral_variance ** 1.5 + 1e-12
#             )

#             cache = self._get_freq_cache(T)
#             low_mask = cache["low_mask"]    # (F,)
#             mid_mask = cache["mid_mask"]
#             high_mask = cache["high_mask"]

#             total_power = psd.sum(axis=1) + 1e-12               # (Nb, C)
#             low_power = psd[:, low_mask, :].sum(axis=1) / total_power   # (Nb, C)
#             mid_power = psd[:, mid_mask, :].sum(axis=1) / total_power
#             high_power = psd[:, high_mask, :].sum(axis=1) / total_power

#             fd_stack = np.stack(
#                 [
#                     mean_freq,
#                     median_freq,
#                     peak_freq,
#                     spectral_entropy,
#                     spectral_variance,
#                     spectral_skewness,
#                     low_power,
#                     mid_power,
#                     high_power,
#                 ],
#                 axis=2,
#             )  # (Nb, C, 9)
#             fd_feats_flat = fd_stack.reshape(Nb, -1)  # (Nb, F_fd)

#             # ------------------- TF (time-frequency energy) -------------------
#             # Просто цикл по фильтрам и жесткий NumPy
#             tf_feats_list = []
#             for sos in self.tf_sos_filters:
#                 try:
#                     filtered = signal.sosfiltfilt(sos, Xb, axis=1)   # (Nb, T, C)
#                     energy = np.sum(filtered ** 2, axis=1)           # (Nb, C)
#                     tf_feats_list.append(energy)
#                 except Exception:
#                     tf_feats_list.append(np.zeros((Nb, C), dtype=float))
#             if tf_feats_list:
#                 tf_feats = np.stack(tf_feats_list, axis=2)    # (Nb, C, n_bands)
#                 tf_feats_flat = tf_feats.reshape(Nb, -1)      # (Nb, F_tf)
#             else:
#                 tf_feats_flat = np.zeros((Nb, 0), dtype=float)

#             # ------------------- Complexity (Hjorth + entropy) -------------------
#             # Для простоты оставляем по-канально в цикле (узкое место, но уже быстрее,
#             # чем полный цикл по окнам).
#             comp_feats = []
#             for i in range(Nb):
#                 feats_win = self._extract_complexity(Xb[i])
#                 comp_feats.append(feats_win)
#             comp_feats = np.stack(comp_feats, axis=0)   # (Nb, F_comp)

#             # ------------------- Cross-channel -------------------
#             cross_feats = []
#             for i in range(Nb):
#                 feats_win = self._extract_cross_channel(Xb[i])
#                 cross_feats.append(feats_win)
#             cross_feats = np.stack(cross_feats, axis=0)  # (Nb, F_cross)

#             # ------------------- Конкатенация -------------------
#             feats_batch = np.concatenate(
#                 [td_feats_flat, fd_feats_flat, tf_feats_flat, comp_feats, cross_feats],
#                 axis=1,
#             )  # (Nb, F_total)

#             # Обработка NaN/Inf
#             if not np.all(np.isfinite(feats_batch)):
#                 if self.logger:
#                     self.logger.error(
#                         "[PowerfulEMGFeatures.extract_batch] Detected NaN/Inf in features. "
#                         "Replacing with 0."
#                     )
#                 feats_batch = np.nan_to_num(feats_batch, nan=0.0, posinf=0.0, neginf=0.0)

#             feats_all.append(feats_batch.astype(np.float32))

#         return np.concatenate(feats_all, axis=0)

#     def extract(self, window: np.ndarray) -> np.ndarray:
#         """
#         Извлечь все признаки из одного окна.

#         Args:
#             window: (T, C) окно EMG

#         Returns:
#             features: (F,) вектор признаков (float32)
#         """
#         if window.ndim != 2:
#             raise ValueError(f"Expected window shape (T, C), got {window.shape}")

#         T, C = window.shape

#         # 1. Multi-scale TD
#         td_features = self._extract_multiscale_td(window)

#         # 2. Frequency-domain
#         fd_features = self._extract_frequency(window)

#         # 3. Time-Frequency
#         tf_features = self._extract_time_frequency(window)

#         # 4. Complexity
#         complexity_features = self._extract_complexity(window)

#         # 5. Cross-channel
#         cross_features = self._extract_cross_channel(window)

#         all_features = np.concatenate(
#             [
#                 td_features,
#                 fd_features,
#                 tf_features,
#                 complexity_features,
#                 cross_features,
#             ],
#             axis=0,
#         )


#         if not np.all(np.isfinite(all_features)):
#             if self.logger:
#                 self.logger.error(
#                     "[PowerfulEMGFeatures] Detected NaN/Inf in feature vector. "
#                     f"Replacing with 0. "
#                     f"Stats: min={np.nanmin(all_features)}, max={np.nanmax(all_features)}"
#                 )
#             # заменяем NaN/Inf на 0, чтобы не валить обучение
#             all_features = np.nan_to_num(all_features, nan=0.0, posinf=0.0, neginf=0.0)

#         return all_features.astype(np.float32)

#     # ------------------------------------------------------------------ #
#     #                    1. Multi-scale time-domain                      #
#     # ------------------------------------------------------------------ #

#     def _extract_multiscale_td(self, window: np.ndarray) -> np.ndarray:
#         """
#         Многомасштабные TD‑признаки: считаем по 3 сегментам (начало/середина/конец)
#         векторизовано по каналам.

#         Для каждой части и каждого канала: MAV, RMS, STD, WL, ZC.
#         """
#         T, C = window.shape
#         third = T // 3
#         parts = [
#             window[:third, :],
#             window[third:2 * third, :],
#             window[2 * third :, :],
#         ]

#         feats_parts = []
#         for part in parts:
#             L = part.shape[0]
#             if L == 0:
#                 continue

#             # MAV, RMS, STD
#             mav = np.mean(np.abs(part), axis=0)            # (C,)
#             rms = np.sqrt(np.mean(part ** 2, axis=0))      # (C,)
#             std = np.std(part, axis=0)                     # (C,)

#             if L > 1:
#                 diff = np.diff(part, axis=0)               # (L-1, C)
#                 wl = np.sum(np.abs(diff), axis=0)          # (C,)

#                 signs = np.sign(part)
#                 zc = (np.diff(signs, axis=0) != 0).sum(axis=0) / (L + 1e-12)  # (C,)
#             else:
#                 wl = np.zeros(C, dtype=float)
#                 zc = np.zeros(C, dtype=float)

#             feats_part = np.stack([mav, rms, std, wl, zc], axis=1)  # (C, 5)
#             feats_parts.append(feats_part)

#         if not feats_parts:
#             return np.zeros(0, dtype=float)

#         feats = np.concatenate(feats_parts, axis=1)  # (C, 5 * n_parts)
#         return feats.ravel()

#     # ------------------------------------------------------------------ #
#     #                        2. Frequency-domain                         #
#     # ------------------------------------------------------------------ #

#     def _get_freq_cache(self, T: int):
#         """Кэшируем freqs и маски частотных полос для данной длины окна."""
#         if T in self._freq_cache:
#             return self._freq_cache[T]

#         freqs = rfftfreq(T, 1 / self.fs)  # (F,)

#         low_mask = (freqs >= 20) & (freqs < 80)
#         mid_mask = (freqs >= 80) & (freqs < 150)
#         high_mask = (freqs >= 150) & (freqs < 450)

#         cache = {
#             "freqs": freqs,
#             "low_mask": low_mask,
#             "mid_mask": mid_mask,
#             "high_mask": high_mask,
#         }
#         self._freq_cache[T] = cache
#         return cache

#     def _extract_frequency(self, window: np.ndarray) -> np.ndarray:
#         T, C = window.shape
#         cache = self._get_freq_cache(T)
#         freqs = cache["freqs"]
#         low_mask = cache["low_mask"]
#         mid_mask = cache["mid_mask"]
#         high_mask = cache["high_mask"]

#         # ---- GPU‑ветка (torch) ----
#         if self.use_torch and self.device == "cuda":
#             x = torch.as_tensor(window, dtype=torch.float32, device=self.device)  # (T, C)

#             # FFT по времени
#             fft_vals = torch.fft.rfft(x, dim=0)         # (F, C)
#             psd = (fft_vals.abs() ** 2)                 # (F, C)

#             psd_sum = psd.sum(dim=0, keepdim=True) + 1e-12
#             psd_norm = psd / psd_sum                    # (F, C)

#             freqs_t = torch.as_tensor(freqs, dtype=torch.float32, device=self.device)

#             # mean freq
#             mean_freq = (freqs_t[:, None] * psd_norm).sum(dim=0)  # (C,)

#             # median freq
#             cumsum_psd = torch.cumsum(psd_norm, dim=0)            # (F, C)
#             mask = cumsum_psd >= 0.5                              # bool (F, C)
#             # argmax по float маске — первый True станет idx медианы
#             median_idx = mask.float().argmax(dim=0)               # (C,)
#             median_freq = freqs_t[median_idx]                     # (C,)

#             # peak freq
#             peak_idx = psd.argmax(dim=0)                          # (C,)
#             peak_freq = freqs_t[peak_idx]                         # (C,)

#             # spectral entropy
#             spectral_entropy = -(psd_norm * torch.log(psd_norm + 1e-12)).sum(dim=0)  # (C,)

#             # variance & skewness
#             freq_centered = freqs_t[:, None] - mean_freq[None, :]  # (F, C)
#             spectral_variance = (psd_norm * (freq_centered ** 2)).sum(dim=0)         # (C,)
#             spectral_skewness = (psd_norm * (freq_centered ** 3)).sum(dim=0) / (
#                 spectral_variance ** 1.5 + 1e-12
#             )

#             # band powers
#             low_idx = torch.as_tensor(np.where(low_mask)[0], dtype=torch.long, device=self.device)
#             mid_idx = torch.as_tensor(np.where(mid_mask)[0], dtype=torch.long, device=self.device)
#             high_idx = torch.as_tensor(np.where(high_mask)[0], dtype=torch.long, device=self.device)

#             total_power = psd.sum(dim=0) + 1e-12
#             low_power = psd[low_idx].sum(dim=0) / total_power
#             mid_power = psd[mid_idx].sum(dim=0) / total_power
#             high_power = psd[high_idx].sum(dim=0) / total_power

#             features_per_channel = torch.stack(
#                 [
#                     mean_freq,
#                     median_freq,
#                     peak_freq,
#                     spectral_entropy,
#                     spectral_variance,
#                     spectral_skewness,
#                     low_power,
#                     mid_power,
#                     high_power,
#                 ],
#                 dim=1,
#             )  # (C, 9)

#             return features_per_channel.detach().cpu().numpy().ravel()

#         # ---- CPU‑ветка (numpy) ----
#         fft_vals = rfft(window, axis=0)  # (F, C)
#         psd = np.abs(fft_vals) ** 2      # (F, C)

#         psd_sum = psd.sum(axis=0, keepdims=True) + 1e-12
#         psd_norm = psd / psd_sum

#         mean_freq = (freqs[:, None] * psd_norm).sum(axis=0)  # (C,)

#         cumsum_psd = np.cumsum(psd_norm, axis=0)             # (F, C)
#         median_idx = (cumsum_psd >= 0.5).argmax(axis=0)
#         median_freq = freqs[median_idx]                      # (C,)

#         peak_idx = psd.argmax(axis=0)
#         peak_freq = freqs[peak_idx]                          # (C,)

#         spectral_entropy = -(psd_norm * np.log(psd_norm + 1e-12)).sum(axis=0)

#         freq_centered = freqs[:, None] - mean_freq[None, :]  # (F, C)
#         spectral_variance = (psd_norm * (freq_centered ** 2)).sum(axis=0)
#         spectral_skewness = (psd_norm * (freq_centered ** 3)).sum(axis=0) / (
#             spectral_variance ** 1.5 + 1e-12
#         )

#         total_power = psd.sum(axis=0) + 1e-12
#         low_power = psd[low_mask].sum(axis=0) / total_power
#         mid_power = psd[mid_mask].sum(axis=0) / total_power
#         high_power = psd[high_mask].sum(axis=0) / total_power

#         features_per_channel = np.stack(
#             [
#                 mean_freq,
#                 median_freq,
#                 peak_freq,
#                 spectral_entropy,
#                 spectral_variance,
#                 spectral_skewness,
#                 low_power,
#                 mid_power,
#                 high_power,
#             ],
#             axis=1,
#         )  # (C, 9)

#         return features_per_channel.ravel()

#     # ------------------------------------------------------------------ #
#     #                       3. Time-Frequency block                      #
#     # ------------------------------------------------------------------ #

#     def _extract_time_frequency(self, window: np.ndarray) -> np.ndarray:
#         """
#         Time‑frequency признаки: энергия в 4 полосах.
#         Фильтруем СРАЗУ все каналы для каждой полосы (sosfiltfilt).
#         """
#         T, C = window.shape
#         feats = []

#         for sos in self.tf_sos_filters:
#             try:
#                 # (T, C) -> (T, C) фильтрация по оси 0
#                 filtered = signal.sosfiltfilt(sos, window, axis=0)
#                 energy = np.sum(filtered ** 2, axis=0)  # (C,)
#                 feats.extend(energy.tolist())
#             except Exception:
#                 feats.extend([0.0] * C)

#         return np.asarray(feats, dtype=float)

#     # ------------------------------------------------------------------ #
#     #                         4. Complexity block                        #
#     # ------------------------------------------------------------------ #

#     def _extract_complexity(self, window: np.ndarray) -> np.ndarray:
#         """
#         Сложность сигнала по каналам:
#         - Sample Entropy (на даунсэмплированном сигнале, опционально)
#         - Hjorth mobility + complexity
#         """
#         T, C = window.shape
#         features = []

#         for ch in range(C):
#             sig = window[:, ch]

#             # Hjorth параметры (на полном сигнале)
#             hj_mob, hj_comp = self._hjorth_parameters(sig)

#             if self.use_entropy:
#                 # Даунсэмплинг для ускорения энтропий
#                 sig_ds = sig[:: self.entropy_downsample]
#                 try:
#                     samp_ent = self._sample_entropy_fast(sig_ds, m=2, r=0.2)
#                 except Exception:
#                     samp_ent = 0.0
#                 # Можно добавить approximate entropy аналогично, но это удваивает время.
#                 approx_ent = samp_ent  # для сохранения "структуры" признаков
#                 features.extend([samp_ent, approx_ent, hj_mob, hj_comp])
#             else:
#                 features.extend([hj_mob, hj_comp])

#         return np.asarray(features, dtype=float)

#     def _hjorth_parameters(self, sig: np.ndarray):
#         """
#         Hjorth mobility и complexity.
#         """
#         if sig.size < 3:
#             return 0.0, 0.0

#         d1 = np.diff(sig)
#         d2 = np.diff(d1)

#         var_sig = np.var(sig)
#         var_d1 = np.var(d1)
#         var_d2 = np.var(d2)

#         mobility = np.sqrt(var_d1 / (var_sig + 1e-12))
#         complexity = np.sqrt(var_d2 / (var_d1 + 1e-12)) / (mobility + 1e-12)
#         return float(mobility), float(complexity)

#     def _sample_entropy_fast(self, sig: np.ndarray, m: int = 2, r: float = 0.2) -> float:
#         """
#         Векторизованная (быстрая) оценка Sample Entropy.
#         Работает на даунсэмплированном сигнале (обычно N ~ 200–300).
#         """
#         N = sig.size
#         if N <= m + 2:
#             return 0.0

#         std = np.std(sig)
#         if std < 1e-12:
#             return 0.0
#         r_abs = r * std

#         def _phi(order: int) -> float:
#             # Встроенные вектораизованные шаблоны: (N-order+1, order)
#             M = N - order + 1
#             x = np.lib.stride_tricks.sliding_window_view(sig, window_shape=order)
#             # x: (M, order)
#             # Вычисляем Chebyshev расстояния между всеми парами шаблонов
#             # dist[i,j] = max_k |x[i,k] - x[j,k]|
#             diff = np.abs(x[None, :, :] - x[:, None, :])      # (M, M, order)
#             dist = np.max(diff, axis=2)                       # (M, M)

#             # Считаем совпадения ≤ r_abs (вычитаем самих себя)
#             C = (dist <= r_abs).sum(axis=1) - 1               # (M,)
#             return C.sum() / (M * (M - 1) + 1e-12)

#         try:
#             phi_m = _phi(m)
#             phi_m1 = _phi(m + 1)
#             if phi_m1 <= 0 or phi_m <= 0:
#                 return 0.0
#             return float(-np.log(phi_m1 / phi_m))
#         except Exception:
#             return 0.0

#     # ------------------------------------------------------------------ #
#     #                       5. Cross-channel block                       #
#     # ------------------------------------------------------------------ #

#     def _extract_cross_channel(self, window: np.ndarray) -> np.ndarray:
#         """
#         Межканальные признаки:
#         - корреляция соседних каналов
#         - корреляция "противоположных" (если 8 каналов)
#         - индекс доминирующего канала (по энергии)
#         - отношение max/min энергии канала
#         """
#         T, C = window.shape
#         feats = []

#         # Корреляции соседних
#         for ch in range(C - 1):
#             x = window[:, ch]
#             y = window[:, ch + 1]
#             if np.std(x) < 1e-12 or np.std(y) < 1e-12:
#                 corr = 0.0
#             else:
#                 corr = np.corrcoef(x, y)[0, 1]
#             feats.append(float(corr))

#         # Корреляции "противоположных" при 8 каналах
#         if C == 8:
#             for i in range(4):
#                 x = window[:, i]
#                 y = window[:, i + 4]
#                 if np.std(x) < 1e-12 or np.std(y) < 1e-12:
#                     corr = 0.0
#                 else:
#                     corr = np.corrcoef(x, y)[0, 1]
#                 feats.append(float(corr))

#         # Доминирующий канал
#         channel_energies = np.sum(window ** 2, axis=0)  # (C,)
#         dominant_ch = int(np.argmax(channel_energies))
#         feats.append(dominant_ch / max(1, C))

#         # max/min energy ratio
#         max_energy = float(channel_energies.max())
#         min_energy = float(channel_energies.min() + 1e-12)
#         feats.append(max_energy / min_energy)

#         return np.asarray(feats, dtype=float)


# # ---------------------------------------------------------------------- #
# #             Обновлённый враппер для интеграции в пайплайн            #
# # ---------------------------------------------------------------------- #

# from joblib import Parallel, delayed
# import numpy as np
# from typing import Optional
# import logging


# class PowerfulFeatureExtractor:
#     """
#     Wrapper для интеграции в pipeline.
#     Совместим с API HandcraftedFeatureExtractor.
#     """

#     def __init__(
#         self,
#         sampling_rate: int = 2000,
#         logger: Optional[logging.Logger] = None,
#         feature_set: str = "powerful",
#         n_jobs: int = -1,
#         use_torch: bool = False,
#         device: Optional[str] = None,
#         use_entropy: bool = True,
#         entropy_downsample: int = 4,
#     ):
#         self.fs = sampling_rate
#         self.logger = logger
#         self.feature_set = feature_set
#         self.n_jobs = n_jobs

#         self.extractor = PowerfulEMGFeatures(
#             sampling_rate=sampling_rate,
#             logger=logger,
#             use_torch=use_torch,
#             device=device,
#             use_entropy=use_entropy,
#             entropy_downsample=entropy_downsample,
#         )

#     def transform(self, X: np.ndarray) -> np.ndarray:
#         """
#         X: (N, T, C) окна
#         Returns:
#             features: (N, F)
#         """
#         if X.ndim != 3:
#             raise ValueError(f"Expected X shape (N, T, C), got {X.shape}")

#         N = len(X)
#         if self.logger:
#             self.logger.info(
#                 f"[PowerfulFeatureExtractor] Extracting powerful features "
#                 f"from {N} windows (n_jobs={self.n_jobs})..."
#             )

#         if self.n_jobs is None or self.n_jobs == 1:
#             feats_list = [self.extractor.extract(w) for w in X]
#         else:
#             feats_list = Parallel(n_jobs=self.n_jobs)(
#                 delayed(self.extractor.extract)(X[i]) for i in range(N)
#             )

#         features = np.stack(feats_list, axis=0).astype(np.float32)

#         if hasattr(self.extractor, "extract_batch"):
#             # Можно передавать batch_size, если нужно ограничить память
#             features = self.extractor.extract_batch(X)
#         else:
#             # Старый режим (на всякий случай)
#             if self.n_jobs is None or self.n_jobs == 1:
#                 feats_list = [self.extractor.extract(w) for w in X]
#             else:
#                 feats_list = Parallel(n_jobs=self.n_jobs)(
#                     delayed(self.extractor.extract)(X[i]) for i in range(N)
#                 )
#             features = np.stack(feats_list, axis=0).astype(np.float32)

#         if self.logger:
#             self.logger.info(
#                 f"[PowerfulFeatureExtractor] Done. Shape: {features.shape} "
#                 f"({features.shape[1]} features per window)"
#             )
#         return features

# ===================================================
import logging
from typing import Optional, Dict
import numpy as np
from scipy import signal

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None


class PowerfulEMGFeatures:
    """
    Оптимизированный извлекатель EMG-признаков с полной GPU поддержкой.
    """

    def __init__(
        self,
        sampling_rate: int = 2000,
        logger: Optional[logging.Logger] = None,
        use_torch: bool = False,
        device: Optional[str] = None,
        use_entropy: bool = True,
        entropy_downsample: int = 4,
    ):
        self.fs = sampling_rate
        self.logger = logger

        # Torch / GPU настройки
        self.use_torch = bool(use_torch and (torch is not None))
        if self.use_torch:
            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
        else:
            self.device = "cpu"

        if self.logger:
            self.logger.info(
                f"[PowerfulEMGFeatures] use_torch={self.use_torch}, device={self.device}"
            )

        # Time-frequency фильтры
        nyquist = 0.5 * self.fs
        self.tf_bands = [
            (20, 50),      # Low
            (50, 100),     # Mid-Low
            (100, 200),    # Mid-High
            (200, 450),    # High
        ]
        
        # Предвычисляем SOS фильтры для CPU
        self.tf_sos_filters = []
        for low, high in self.tf_bands:
            high_ = min(high, 0.45 * self.fs)
            wp = [low / nyquist, high_ / nyquist]
            sos = signal.butter(4, wp, btype="band", output="sos")
            self.tf_sos_filters.append(sos)

        # Для GPU: конвертируем SOS в FIR фильтры
        if self.use_torch and self.device == "cuda":
            self._prepare_gpu_filters()

        self.use_entropy = use_entropy
        self.entropy_downsample = max(1, int(entropy_downsample))
        self._freq_cache: Dict[int, Dict] = {}

    def _prepare_gpu_filters(self):
        """Подготовка FIR фильтров для GPU (альтернатива SOS)."""
        # Создаём FIR фильтры из band-pass спецификаций
        self.gpu_filters = []
        numtaps = 101  # Длина FIR фильтра
        
        for low, high in self.tf_bands:
            high_ = min(high, 0.45 * self.fs)
            # Создаём FIR коэффициенты
            h = signal.firwin(
                numtaps, 
                [low, high_], 
                pass_zero=False, 
                fs=self.fs
            )
            h_tensor = torch.tensor(h, dtype=torch.float32, device=self.device)
            self.gpu_filters.append(h_tensor)

    # ================================================================
    #                    БАТЧЕВАЯ GPU ОБРАБОТКА
    # ================================================================

    def extract_batch_gpu(self, X: np.ndarray) -> np.ndarray:
        """
        Полностью GPU-батчевая обработка.
        X: (N, T, C) на CPU
        Returns: (N, F) на CPU
        """
        if not self.use_torch or self.device == "cpu":
            return self.extract_batch(X)

        N, T, C = X.shape
        
        # Переносим данные на GPU ОДИН РАЗ
        X_gpu = torch.tensor(X, dtype=torch.float32, device=self.device)  # (N, T, C)

        # 1. Time-Domain признаки (GPU)
        td_feats = self._extract_multiscale_td_gpu(X_gpu)  # (N, F_td)

        # 2. Frequency-Domain признаки (GPU)
        fd_feats = self._extract_frequency_gpu(X_gpu)      # (N, F_fd)

        # 3. Time-Frequency признаки (GPU)
        tf_feats = self._extract_time_frequency_gpu(X_gpu) # (N, F_tf)

        # 4. Complexity + Cross-channel (эти сложнее перенести на GPU)
        # Используем быструю CPU версию для оставшихся признаков
        X_cpu = X  # Уже на CPU
        comp_feats = self._extract_complexity_batch_cpu(X_cpu)      # (N, F_comp)
        cross_feats = self._extract_cross_channel_batch_cpu(X_cpu)  # (N, F_cross)

        # Конкатенация на GPU
        comp_gpu = torch.tensor(comp_feats, dtype=torch.float32, device=self.device)
        cross_gpu = torch.tensor(cross_feats, dtype=torch.float32, device=self.device)

        all_feats_gpu = torch.cat(
            [td_feats, fd_feats, tf_feats, comp_gpu, cross_gpu], 
            dim=1
        )  # (N, F_total)

        # Возврат на CPU
        result = all_feats_gpu.cpu().numpy()

        # Обработка NaN/Inf
        if not np.all(np.isfinite(result)):
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

        return result.astype(np.float32)

    # ----------------------------------------------------------------
    #                   GPU Time-Domain блок
    # ----------------------------------------------------------------

    def _extract_multiscale_td_gpu(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (N, T, C) на GPU
        Returns: (N, F_td) на GPU
        """
        N, T, C = X.shape
        third = T // 3

        parts = [
            X[:, :third, :],
            X[:, third:2*third, :],
            X[:, 2*third:, :],
        ]

        feats_parts = []
        for part in parts:
            L = part.shape[1]
            if L == 0:
                continue

            # MAV, RMS, STD
            mav = torch.mean(torch.abs(part), dim=1)           # (N, C)
            rms = torch.sqrt(torch.mean(part ** 2, dim=1))     # (N, C)
            std = torch.std(part, dim=1)                       # (N, C)

            if L > 1:
                diff = part[:, 1:, :] - part[:, :-1, :]        # (N, L-1, C)
                wl = torch.sum(torch.abs(diff), dim=1)         # (N, C)

                signs = torch.sign(part)
                sign_diff = signs[:, 1:, :] - signs[:, :-1, :]
                zc = (sign_diff != 0).float().sum(dim=1) / (L + 1e-12)  # (N, C)
            else:
                wl = torch.zeros(N, C, device=X.device)
                zc = torch.zeros(N, C, device=X.device)

            feats_part = torch.stack([mav, rms, std, wl, zc], dim=2)  # (N, C, 5)
            feats_parts.append(feats_part)

        if not feats_parts:
            return torch.zeros(N, 0, device=X.device)

        feats = torch.cat(feats_parts, dim=2)  # (N, C, 5*n_parts)
        return feats.reshape(N, -1)            # (N, F_td)

    # ----------------------------------------------------------------
    #                   GPU Frequency-Domain блок
    # ----------------------------------------------------------------

    def _extract_frequency_gpu(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (N, T, C) на GPU
        Returns: (N, F_fd) на GPU
        """
        N, T, C = X.shape

        # FFT по времени
        fft_vals = torch.fft.rfft(X, dim=1)         # (N, F, C)
        psd = torch.abs(fft_vals) ** 2              # (N, F, C)

        psd_sum = psd.sum(dim=1, keepdim=True) + 1e-12
        psd_norm = psd / psd_sum                    # (N, F, C)

        freqs = torch.fft.rfftfreq(T, 1/self.fs, device=X.device)  # (F,)

        # Mean frequency
        mean_freq = (freqs[None, :, None] * psd_norm).sum(dim=1)  # (N, C)

        # Median frequency
        cumsum_psd = torch.cumsum(psd_norm, dim=1)              # (N, F, C)
        median_idx = (cumsum_psd >= 0.5).float().argmax(dim=1)  # (N, C)
        median_freq = freqs[median_idx]                         # (N, C)

        # Peak frequency
        peak_idx = psd.argmax(dim=1)                            # (N, C)
        peak_freq = freqs[peak_idx]                             # (N, C)

        # Spectral entropy
        spectral_entropy = -(psd_norm * torch.log(psd_norm + 1e-12)).sum(dim=1)

        # Variance & skewness
        freq_centered = freqs[None, :, None] - mean_freq[:, None, :]  # (N, F, C)
        spectral_variance = (psd_norm * (freq_centered ** 2)).sum(dim=1)
        spectral_skewness = (psd_norm * (freq_centered ** 3)).sum(dim=1) / (
            spectral_variance ** 1.5 + 1e-12
        )

        # Band powers
        low_mask = (freqs >= 20) & (freqs < 80)
        mid_mask = (freqs >= 80) & (freqs < 150)
        high_mask = (freqs >= 150) & (freqs < 450)

        total_power = psd.sum(dim=1) + 1e-12
        low_power = psd[:, low_mask, :].sum(dim=1) / total_power
        mid_power = psd[:, mid_mask, :].sum(dim=1) / total_power
        high_power = psd[:, high_mask, :].sum(dim=1) / total_power

        # Stack all features
        features = torch.stack([
            mean_freq,
            median_freq,
            peak_freq,
            spectral_entropy,
            spectral_variance,
            spectral_skewness,
            low_power,
            mid_power,
            high_power,
        ], dim=2)  # (N, C, 9)

        return features.reshape(N, -1)  # (N, F_fd)

    # ----------------------------------------------------------------
    #                   GPU Time-Frequency блок
    # ----------------------------------------------------------------

    def _extract_time_frequency_gpu(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (N, T, C) на GPU
        Returns: (N, F_tf) на GPU
        """
        N, T, C = X.shape
        tf_feats = []

        for h in self.gpu_filters:
            # Применяем FIR фильтр через convolution
            # X: (N, T, C) -> (N, C, T) для conv1d
            X_trans = X.permute(0, 2, 1)  # (N, C, T)
            
            # Добавляем batch dimension для фильтра
            h_exp = h.unsqueeze(0).unsqueeze(0)  # (1, 1, numtaps)
            
            # Применяем фильтр к каждому каналу
            filtered_channels = []
            for c in range(C):
                filtered_c = F.conv1d(
                    X_trans[:, c:c+1, :],  # (N, 1, T)
                    h_exp,
                    padding='same'
                )  # (N, 1, T)
                filtered_channels.append(filtered_c)
            
            filtered = torch.cat(filtered_channels, dim=1)  # (N, C, T)
            
            # Энергия
            energy = torch.sum(filtered ** 2, dim=2)  # (N, C)
            tf_feats.append(energy)

        if not tf_feats:
            return torch.zeros(N, 0, device=X.device)

        tf_feats = torch.stack(tf_feats, dim=2)  # (N, C, n_bands)
        return tf_feats.reshape(N, -1)           # (N, F_tf)

    # ----------------------------------------------------------------
    #            Быстрые CPU версии для complexity/cross
    # ----------------------------------------------------------------

    def _extract_complexity_batch_cpu(self, X: np.ndarray) -> np.ndarray:
        """Векторизованная версия complexity для батча на CPU."""
        N, T, C = X.shape
        all_feats = []

        for ch in range(C):
            sig = X[:, :, ch]  # (N, T)

            # Hjorth mobility
            d1 = np.diff(sig, axis=1)  # (N, T-1)
            var_sig = np.var(sig, axis=1)  # (N,)
            var_d1 = np.var(d1, axis=1)    # (N,)
            mobility = np.sqrt(var_d1 / (var_sig + 1e-12))

            # Hjorth complexity
            d2 = np.diff(d1, axis=1)  # (N, T-2)
            var_d2 = np.var(d2, axis=1)  # (N,)
            complexity = np.sqrt(var_d2 / (var_d1 + 1e-12)) / (mobility + 1e-12)

            if self.use_entropy:
                # Упрощённая энтропия (заглушки для совместимости)
                samp_ent = np.zeros(N)
                approx_ent = np.zeros(N)
                # Stack: (N, 4)
                ch_feats = np.stack([samp_ent, approx_ent, mobility, complexity], axis=1)
            else:
                # Stack: (N, 2)
                ch_feats = np.stack([mobility, complexity], axis=1)
            
            all_feats.append(ch_feats)

        # Concatenate по каналам: (N, C*feats_per_channel)
        return np.concatenate(all_feats, axis=1)

    def _extract_cross_channel_batch_cpu(self, X: np.ndarray) -> np.ndarray:
        """Векторизованная версия cross-channel для батча на CPU."""
        N, T, C = X.shape
        feats_list = []

        # Корреляции соседних каналов
        for ch in range(C - 1):
            x = X[:, :, ch]      # (N, T)
            y = X[:, :, ch + 1]  # (N, T)
            
            # Векторизованная корреляция по батчу
            x_centered = x - x.mean(axis=1, keepdims=True)
            y_centered = y - y.mean(axis=1, keepdims=True)
            
            corr = (x_centered * y_centered).sum(axis=1) / (
                np.sqrt((x_centered ** 2).sum(axis=1)) * 
                np.sqrt((y_centered ** 2).sum(axis=1)) + 1e-12
            )
            feats_list.append(corr)

        # Корреляции противоположных (для 8 каналов)
        if C == 8:
            for i in range(4):
                x = X[:, :, i]
                y = X[:, :, i + 4]
                x_centered = x - x.mean(axis=1, keepdims=True)
                y_centered = y - y.mean(axis=1, keepdims=True)
                corr = (x_centered * y_centered).sum(axis=1) / (
                    np.sqrt((x_centered ** 2).sum(axis=1)) * 
                    np.sqrt((y_centered ** 2).sum(axis=1)) + 1e-12
                )
                feats_list.append(corr)

        # Доминирующий канал
        channel_energies = np.sum(X ** 2, axis=1)  # (N, C)
        dominant_ch = np.argmax(channel_energies, axis=1) / max(1, C)
        feats_list.append(dominant_ch)

        # Max/min energy ratio
        max_energy = channel_energies.max(axis=1)
        min_energy = channel_energies.min(axis=1) + 1e-12
        feats_list.append(max_energy / min_energy)

        return np.stack(feats_list, axis=1)  # (N, F_cross)

    # ================================================================
    #                   CPU FALLBACK (старая версия)
    # ================================================================

    def extract_batch(self, X: np.ndarray) -> np.ndarray:
        """CPU версия (полная реализация)."""
        if X.ndim != 3:
            raise ValueError(f"Expected X shape (N, T, C), got {X.shape}")
        
        N, T, C = X.shape

        # ------------------- TD -------------------
        third = T // 3
        parts = [
            X[:, :third, :],
            X[:, third:2 * third, :],
            X[:, 2 * third :, :],
        ]
        td_feats_parts = []
        for part in parts:
            L = part.shape[1]
            if L == 0:
                continue
            mav = np.mean(np.abs(part), axis=1)
            rms = np.sqrt(np.mean(part ** 2, axis=1))
            std = np.std(part, axis=1)
            if L > 1:
                diff = np.diff(part, axis=1)
                wl = np.sum(np.abs(diff), axis=1)
                signs = np.sign(part)
                zc = (np.diff(signs, axis=1) != 0).sum(axis=1) / (L + 1e-12)
            else:
                wl = np.zeros((N, C), dtype=float)
                zc = np.zeros((N, C), dtype=float)
            feats_part = np.stack([mav, rms, std, wl, zc], axis=2)
            td_feats_parts.append(feats_part)
        
        if td_feats_parts:
            td_feats = np.concatenate(td_feats_parts, axis=2)
            td_feats_flat = td_feats.reshape(N, -1)
        else:
            td_feats_flat = np.zeros((N, 0), dtype=float)

        # ------------------- FD -------------------
        fft_vals = np.fft.rfft(X, axis=1)
        psd = np.abs(fft_vals) ** 2
        psd_sum = psd.sum(axis=1, keepdims=True) + 1e-12
        psd_norm = psd / psd_sum

        freqs = np.fft.rfftfreq(T, 1 / self.fs)
        mean_freq = (psd_norm * freqs[None, :, None]).sum(axis=1)

        cumsum_psd = np.cumsum(psd_norm, axis=1)
        median_idx = (cumsum_psd >= 0.5).argmax(axis=1)
        median_freq = freqs[median_idx]

        peak_idx = psd.argmax(axis=1)
        peak_freq = freqs[peak_idx]

        spectral_entropy = -(psd_norm * np.log(psd_norm + 1e-12)).sum(axis=1)

        freq_centered = freqs[None, :, None] - mean_freq[:, None, :]
        spectral_variance = (psd_norm * (freq_centered ** 2)).sum(axis=1)
        spectral_skewness = (psd_norm * (freq_centered ** 3)).sum(axis=1) / (
            spectral_variance ** 1.5 + 1e-12
        )

        low_mask = (freqs >= 20) & (freqs < 80)
        mid_mask = (freqs >= 80) & (freqs < 150)
        high_mask = (freqs >= 150) & (freqs < 450)

        total_power = psd.sum(axis=1) + 1e-12
        low_power = psd[:, low_mask, :].sum(axis=1) / total_power
        mid_power = psd[:, mid_mask, :].sum(axis=1) / total_power
        high_power = psd[:, high_mask, :].sum(axis=1) / total_power

        fd_stack = np.stack([
            mean_freq, median_freq, peak_freq, spectral_entropy,
            spectral_variance, spectral_skewness,
            low_power, mid_power, high_power,
        ], axis=2)
        fd_feats_flat = fd_stack.reshape(N, -1)

        # ------------------- TF -------------------
        tf_feats_list = []
        for sos in self.tf_sos_filters:
            try:
                filtered = signal.sosfiltfilt(sos, X, axis=1)
                energy = np.sum(filtered ** 2, axis=1)
                tf_feats_list.append(energy)
            except Exception:
                tf_feats_list.append(np.zeros((N, C), dtype=float))
        
        if tf_feats_list:
            tf_feats = np.stack(tf_feats_list, axis=2)
            tf_feats_flat = tf_feats.reshape(N, -1)
        else:
            tf_feats_flat = np.zeros((N, 0), dtype=float)

        # ------------------- Complexity + Cross -------------------
        comp_feats = self._extract_complexity_batch_cpu(X)
        cross_feats = self._extract_cross_channel_batch_cpu(X)

        # ------------------- Concat -------------------
        feats_batch = np.concatenate([
            td_feats_flat, fd_feats_flat, tf_feats_flat, 
            comp_feats, cross_feats
        ], axis=1)

        if not np.all(np.isfinite(feats_batch)):
            feats_batch = np.nan_to_num(feats_batch, nan=0.0, posinf=0.0, neginf=0.0)

        return feats_batch.astype(np.float32)

    def extract(self, window: np.ndarray) -> np.ndarray:
        """Одиночное окно (для совместимости)."""
        # Используем батчевую версию с N=1
        X = window[np.newaxis, :, :]  # (1, T, C)
        if self.use_torch and self.device == "cuda":
            feats = self.extract_batch_gpu(X)
        else:
            feats = self.extract_batch(X)
        return feats[0]


# ================================================================
#                         ОБНОВЛЁННЫЙ ВРАППЕР
# ================================================================

from joblib import Parallel, delayed


class PowerfulFeatureExtractor:
    """Wrapper с автоматическим выбором GPU/CPU режима."""

    def __init__(
        self,
        sampling_rate: int = 2000,
        logger: Optional[logging.Logger] = None,
        feature_set: str = "powerful",
        n_jobs: int = -1,
        use_torch: bool = False,
        device: Optional[str] = None,
        use_entropy: bool = True,
        entropy_downsample: int = 4,
        gpu_batch_size: int = 4096,  # Размер батча для GPU
    ):
        self.fs = sampling_rate
        self.logger = logger
        self.feature_set = feature_set
        self.n_jobs = n_jobs
        self.gpu_batch_size = gpu_batch_size

        self.extractor = PowerfulEMGFeatures(
            sampling_rate=sampling_rate,
            logger=logger,
            use_torch=use_torch,
            device=device,
            use_entropy=use_entropy,
            entropy_downsample=entropy_downsample,
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        X: (N, T, C)
        Returns: (N, F)
        """
        if X.ndim != 3:
            raise ValueError(f"Expected X shape (N, T, C), got {X.shape}")

        N = len(X)
        
        # Если GPU доступна - используем батчевую обработку
        if self.extractor.use_torch and self.extractor.device == "cuda":
            if self.logger:
                self.logger.info(
                    f"[PowerfulFeatureExtractor] Using GPU batch processing "
                    f"for {N} windows (batch_size={self.gpu_batch_size})"
                )
            
            feats_list = []
            for i in range(0, N, self.gpu_batch_size):
                batch = X[i:i + self.gpu_batch_size]
                feats_batch = self.extractor.extract_batch_gpu(batch)
                feats_list.append(feats_batch)
                
                if self.logger and (i // self.gpu_batch_size) % 10 == 0:
                    progress = min(100, (i + self.gpu_batch_size) / N * 100)
                    self.logger.info(f"  Progress: {progress:.1f}%")
            
            features = np.concatenate(feats_list, axis=0)
        else:
            # CPU версия
            if self.logger:
                self.logger.info(
                    f"[PowerfulFeatureExtractor] Using CPU (n_jobs={self.n_jobs})"
                )
            features = self.extractor.extract_batch(X)

        if self.logger:
            self.logger.info(
                f"[PowerfulFeatureExtractor] Done. Shape: {features.shape}"
            )
        
        return features