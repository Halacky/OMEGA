"""
КРИТИЧНЫЕ ПРОВЕРКИ КАЧЕСТВА ДАННЫХ
Запусти этот скрипт ПЕРЕД любыми улучшениями моделей
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import logging

class DataQualityDiagnostic:
    """Диагностика качества EMG данных"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.issues = []
    
    def check_all(self, splits: dict, class_names: dict):
        """Запуск всех проверок"""
        self.logger.info("="*80)
        self.logger.info("ДИАГНОСТИКА КАЧЕСТВА ДАННЫХ")
        self.logger.info("="*80)
        
        X_train = splits['train']
        
        # Критичные проверки
        self.check_1_channel_correlation(X_train)
        self.check_2_class_separability(splits, class_names)
        self.check_3_signal_quality(X_train)
        self.check_4_channel_placement_consistency(splits)
        self.check_5_rest_vs_gesture_separation(splits)
        
        # Итоговый отчет
        self.print_summary()
        
        return self.issues
    
    def check_1_channel_correlation(self, X_train: dict):
        """
        ПРОВЕРКА 1: Корреляция между каналами
        ПРОБЛЕМА: Если все каналы сильно коррелируют (>0.9), 
        значит они несут одинаковую информацию - модель не может различить жесты
        """
        self.logger.info("\n[CHECK 1] Корреляция между каналами")
        
        all_data = []
        for gid, windows in X_train.items():
            if len(windows) > 0:
                # Усредняем по времени: (N, T, C) -> (N, C)
                all_data.append(windows.mean(axis=1))
        
        if not all_data:
            self.logger.warning("Нет данных для анализа!")
            return
        
        data = np.concatenate(all_data, axis=0)  # (N_total, C)
        
        # Корреляционная матрица между каналами
        corr_matrix = np.corrcoef(data.T)  # (C, C)
        
        # Среднее значение вне диагонали
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        avg_corr = np.abs(corr_matrix[mask]).mean()
        max_corr = np.abs(corr_matrix[mask]).max()
        
        self.logger.info(f"Средняя корреляция между каналами: {avg_corr:.3f}")
        self.logger.info(f"Максимальная корреляция: {max_corr:.3f}")
        
        if avg_corr > 0.85:
            issue = "❌ КРИТИЧНО: Каналы сильно коррелируют! Возможно неправильное размещение электродов."
            self.logger.error(issue)
            self.issues.append(issue)
        elif avg_corr > 0.7:
            issue = "⚠️ ВНИМАНИЕ: Высокая корреляция каналов. Проверь размещение браслета."
            self.logger.warning(issue)
            self.issues.append(issue)
        else:
            self.logger.info("✓ Корреляция каналов в норме")
        
        # Визуализация
        plt.figure(figsize=(10, 8))
        plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label='Correlation')
        plt.title(f'Channel Correlation Matrix\nAvg={avg_corr:.3f}, Max={max_corr:.3f}')
        plt.xlabel('Channel')
        plt.ylabel('Channel')
        plt.tight_layout()
        plt.savefig('diagnostic_channel_correlation.png', dpi=150)
        plt.close()
    
    def check_2_class_separability(self, splits: dict, class_names: dict):
        """
        ПРОВЕРКА 2: Разделимость классов
        ПРОБЛЕМА: Если классы перекрываются в пространстве признаков,
        никакая модель их не различит
        """
        self.logger.info("\n[CHECK 2] Разделимость классов")
        
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.metrics import silhouette_score
        
        X_train = splits['train']
        
        # Подготовка данных
        X_list, y_list = [], []
        class_ids = sorted(X_train.keys())
        
        for i, gid in enumerate(class_ids):
            windows = X_train[gid]
            if len(windows) > 0:
                # Простые признаки: среднее по времени
                features = windows.mean(axis=1)  # (N, C)
                X_list.append(features)
                y_list.append(np.full(len(features), i))
        
        if len(X_list) < 2:
            self.logger.warning("Недостаточно классов для анализа")
            return
        
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        
        # Silhouette Score (-1 до 1, выше = лучше разделимость)
        silhouette = silhouette_score(X, y, metric='euclidean', sample_size=min(5000, len(X)))
        self.logger.info(f"Silhouette Score: {silhouette:.3f}")
        
        if silhouette < 0.1:
            issue = "❌ КРИТИЧНО: Классы НЕ разделимы в пространстве признаков!"
            self.logger.error(issue)
            self.issues.append(issue)
        elif silhouette < 0.3:
            issue = "⚠️ ВНИМАНИЕ: Слабая разделимость классов"
            self.logger.warning(issue)
            self.issues.append(issue)
        else:
            self.logger.info("✓ Разделимость классов приемлемая")
        
        # LDA для визуализации
        if len(class_ids) > 2:
            lda = LinearDiscriminantAnalysis(n_components=2)
            X_lda = lda.fit_transform(X, y)
            
            plt.figure(figsize=(12, 8))
            for i, gid in enumerate(class_ids):
                mask = y == i
                label = class_names.get(gid, f"Class {gid}")
                plt.scatter(X_lda[mask, 0], X_lda[mask, 1], 
                           alpha=0.3, s=10, label=label)
            
            plt.xlabel('LDA Component 1')
            plt.ylabel('LDA Component 2')
            plt.title(f'Class Separability (Silhouette={silhouette:.3f})')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig('diagnostic_class_separability.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    def check_3_signal_quality(self, X_train: dict):
        """
        ПРОВЕРКА 3: Качество сигнала
        ПРОБЛЕМА: Низкое SNR, артефакты, насыщение
        """
        self.logger.info("\n[CHECK 3] Качество сигнала")
        
        all_windows = []
        for windows in X_train.values():
            if len(windows) > 0:
                all_windows.append(windows)
        
        if not all_windows:
            return
        
        all_windows = np.concatenate(all_windows, axis=0)  # (N, T, C)
        
        # 1. Проверка на насыщение (сигнал касается границ)
        signal_std = all_windows.std()
        signal_range = all_windows.max() - all_windows.min()
        
        # Считаем, что сигнал нормализован в [-3, 3] стандартных отклонений
        saturation_threshold = 2.8 * signal_std
        saturated_samples = np.sum(np.abs(all_windows) > saturation_threshold)
        saturation_ratio = saturated_samples / all_windows.size
        
        self.logger.info(f"Доля насыщенных отсчетов: {saturation_ratio*100:.2f}%")
        
        if saturation_ratio > 0.01:
            issue = "❌ КРИТИЧНО: Сигнал насыщается! Нужна калибровка усиления."
            self.logger.error(issue)
            self.issues.append(issue)
        
        # 2. SNR оценка (простая: отношение стандартных отклонений)
        # Предполагаем, что минимальная активность близка к шуму
        window_stds = all_windows.std(axis=(1, 2))  # (N,)
        noise_estimate = np.percentile(window_stds, 10)
        signal_estimate = np.percentile(window_stds, 90)
        
        snr_db = 20 * np.log10(signal_estimate / (noise_estimate + 1e-8))
        self.logger.info(f"Оценка SNR: {snr_db:.1f} dB")
        
        if snr_db < 10:
            issue = "❌ КРИТИЧНО: Очень низкий SNR! Проверь контакт электродов."
            self.logger.error(issue)
            self.issues.append(issue)
        elif snr_db < 20:
            issue = "⚠️ ВНИМАНИЕ: Низкий SNR"
            self.logger.warning(issue)
            self.issues.append(issue)
        else:
            self.logger.info("✓ SNR в норме")
    
    def check_4_channel_placement_consistency(self, splits: dict):
        """
        ПРОВЕРКА 4: Консистентность размещения электродов
        ПРОБЛЕМА: Если браслет крутится между сессиями/субъектами,
        cross-subject точность будет низкой
        """
        self.logger.info("\n[CHECK 4] Консистентность размещения электродов")
        
        X_train = splits['train']
        
        # Для каждого класса смотрим, какие каналы активны
        class_channel_activation = {}
        
        for gid, windows in X_train.items():
            if len(windows) > 0:
                # Средняя активность по каналам: (N, T, C) -> (C,)
                channel_activity = windows.std(axis=(0, 1))  # std по N и T
                class_channel_activation[gid] = channel_activity
        
        if len(class_channel_activation) < 2:
            return
        
        # Проверяем, есть ли паттерн активации каналов
        activations = np.array(list(class_channel_activation.values()))  # (num_classes, C)
        
        # Нормализуем
        activations = activations / (activations.max(axis=1, keepdims=True) + 1e-8)
        
        # Корреляция между паттернами активации разных классов
        corr = np.corrcoef(activations)
        mask = ~np.eye(corr.shape[0], dtype=bool)
        avg_pattern_corr = np.abs(corr[mask]).mean()
        
        self.logger.info(f"Средняя корреляция паттернов активации: {avg_pattern_corr:.3f}")
        
        if avg_pattern_corr > 0.8:
            issue = "❌ КРИТИЧНО: Все жесты активируют одни и те же каналы!"
            self.logger.error(issue)
            self.issues.append(issue)
        elif avg_pattern_corr > 0.6:
            issue = "⚠️ ВНИМАНИЕ: Слабая специфичность каналов для разных жестов"
            self.logger.warning(issue)
            self.issues.append(issue)
        else:
            self.logger.info("✓ Паттерны активации различаются между жестами")
        
        # Визуализация
        plt.figure(figsize=(14, 10))
        im = plt.imshow(activations, aspect='auto', cmap='hot', interpolation='nearest')
        plt.colorbar(im, label='Normalized Activation')
        plt.xlabel('Channel')
        plt.ylabel('Gesture Class')
        plt.title('Channel Activation Patterns per Gesture')
        plt.yticks(range(len(class_channel_activation)), 
                  [f"G{gid}" for gid in class_channel_activation.keys()])
        plt.tight_layout()
        plt.savefig('diagnostic_channel_patterns.png', dpi=150)
        plt.close()
    
    def check_5_rest_vs_gesture_separation(self, splits: dict):
        """
        ПРОВЕРКА 5: REST отделим от жестов?
        ПРОБЛЕМА: Если REST неотличим от жестов, значит проблемы с сегментацией
        """
        self.logger.info("\n[CHECK 5] Разделение REST vs Жесты")
        
        X_train = splits['train']
        
        if 0 not in X_train:
            self.logger.info("REST класс не включен в обучение")
            return
        
        rest_windows = X_train[0]
        gesture_windows = []
        
        for gid, windows in X_train.items():
            if gid != 0 and len(windows) > 0:
                gesture_windows.append(windows)
        
        if len(gesture_windows) == 0:
            return
        
        gesture_windows = np.concatenate(gesture_windows, axis=0)
        
        # Простой признак: общая энергия
        rest_energy = (rest_windows ** 2).sum(axis=(1, 2))
        gesture_energy = (gesture_windows ** 2).sum(axis=(1, 2))
        
        # T-test
        t_stat, p_value = stats.ttest_ind(rest_energy, gesture_energy)
        
        self.logger.info(f"T-test p-value (REST vs Gestures): {p_value:.2e}")
        self.logger.info(f"REST mean energy: {rest_energy.mean():.3f}")
        self.logger.info(f"Gesture mean energy: {gesture_energy.mean():.3f}")
        
        if p_value > 0.01:
            issue = "❌ КРИТИЧНО: REST неотличим от жестов! Проблема с сегментацией."
            self.logger.error(issue)
            self.issues.append(issue)
        else:
            self.logger.info("✓ REST отделим от жестов")
        
        # Визуализация
        plt.figure(figsize=(10, 6))
        plt.hist(rest_energy, bins=50, alpha=0.5, label='REST', density=True)
        plt.hist(gesture_energy, bins=50, alpha=0.5, label='Gestures', density=True)
        plt.xlabel('Signal Energy')
        plt.ylabel('Density')
        plt.legend()
        plt.title(f'REST vs Gestures Energy Distribution (p={p_value:.2e})')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('diagnostic_rest_vs_gestures.png', dpi=150)
        plt.close()
    
    def print_summary(self):
        """Итоговый отчет"""
        self.logger.info("\n" + "="*80)
        self.logger.info("ИТОГОВЫЙ ОТЧЕТ")
        self.logger.info("="*80)
        
        if not self.issues:
            self.logger.info("✓ Все проверки пройдены!")
        else:
            self.logger.error(f"Найдено проблем: {len(self.issues)}")
            for i, issue in enumerate(self.issues, 1):
                self.logger.error(f"{i}. {issue}")
        
        self.logger.info("="*80)
        self.logger.info("Все диагностические графики сохранены с префиксом 'diagnostic_'")
        self.logger.info("="*80)

