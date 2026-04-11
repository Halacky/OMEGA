import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Arrow, Circle, FancyArrowPatch
import numpy as np
from matplotlib.path import Path

# Создаем фигуру с двумя панелями
fig = plt.figure(figsize=(18, 12))

# Панель 1: Конвейер обработки данных
ax1 = plt.subplot2grid((2, 1), (0, 0), rowspan=1, colspan=1)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis('off')
ax1.set_title('Конвейер обработки EMG сигналов NinaPro DB2', fontsize=14, fontweight='bold', pad=20)

# Панель 2: Схема LOSO
ax2 = plt.subplot2grid((2, 1), (1, 0), rowspan=1, colspan=1)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')
ax2.set_title('Схема кросс-субъектной оценки (Leave-One-Subject-Out)', fontsize=14, fontweight='bold', pad=20)

# Цвета для различных блоков
colors = {
    'raw': '#FF6B6B',
    'preprocess': '#4ECDC4',
    'window': '#45B7D1',
    'deep': '#96CEB4',
    'feature': '#FFEAA7',
    'classical': '#DDA0DD',
    'subject': '#A2D5F2',
    'train': '#FFD93D',
    'test': '#FF6B6B',
    'val': '#6BCF7F'
}

# ==================== ПАНЕЛЬ 1: Конвейер обработки ====================

# Блок 1: Сырые данные
raw_box = FancyBboxPatch((0.05, 0.7), 0.15, 0.2, boxstyle="round,pad=0.05",
                        facecolor=colors['raw'], edgecolor='black', linewidth=2)
ax1.add_patch(raw_box)
ax1.text(0.125, 0.8, 'Сырые данные\nNinaPro DB2', ha='center', va='center', 
         fontsize=11, fontweight='bold', color='white')

# Стрелка к предобработке
ax1.annotate('', xy=(0.22, 0.8), xytext=(0.05, 0.8),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Блок 2: Предобработка
preprocess_box = FancyBboxPatch((0.23, 0.7), 0.15, 0.2, boxstyle="round,pad=0.05",
                               facecolor=colors['preprocess'], edgecolor='black', linewidth=2)
ax1.add_patch(preprocess_box)
preprocess_text = 'Предобработка:\n• Trend removal\n• Notch 50 Hz\n• Band-pass 20-450 Hz\n• Обработка насыщений\n• Нормализация'
ax1.text(0.305, 0.8, preprocess_text, ha='center', va='center', 
         fontsize=10, fontweight='bold', color='black')

# Стрелка к ветвлению
ax1.annotate('', xy=(0.40, 0.8), xytext=(0.23, 0.8),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Блок 3: Оконование (общий)
window_box = FancyBboxPatch((0.42, 0.7), 0.15, 0.2, boxstyle="round,pad=0.05",
                           facecolor=colors['window'], edgecolor='black', linewidth=2)
ax1.add_patch(window_box)
ax1.text(0.495, 0.8, 'Оконование\nсигнала', ha='center', va='center', 
         fontsize=11, fontweight='bold', color='white')

# Стрелки от оконования к двум ветвям
# Ветвь 1: Глубокое обучение
ax1.annotate('', xy=(0.59, 0.9), xytext=(0.42, 0.9),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
# Ветвь 2: ML
ax1.annotate('', xy=(0.59, 0.7), xytext=(0.42, 0.7),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Ветвь 1: Глубокое обучение
# Блок: Raw windows
raw_window_box = FancyBboxPatch((0.61, 0.85), 0.15, 0.1, boxstyle="round,pad=0.03",
                               facecolor=colors['deep'], edgecolor='black', linewidth=2)
ax1.add_patch(raw_window_box)
ax1.text(0.685, 0.9, 'Raw windows\n(8×T)', ha='center', va='center', 
         fontsize=10, fontweight='bold', color='black')

# Стрелка к моделям DL
ax1.annotate('', xy=(0.78, 0.9), xytext=(0.61, 0.9),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Модели DL
dl_models = ['CNN', 'LSTM', 'GRU', 'TCN', 'Attention\nCNN', 'CNN-LSTM', 
             'BiLSTM', 'BiGRU', 'Multiscale\nCNN']
for i, model in enumerate(dl_models):
    x = 0.80 + (i % 3) * 0.06
    y = 0.95 - (i // 3) * 0.08
    model_box = FancyBboxPatch((x, y), 0.055, 0.06, boxstyle="round,pad=0.01",
                              facecolor=colors['deep'], edgecolor='black', linewidth=1.5)
    ax1.add_patch(model_box)
    ax1.text(x + 0.0275, y + 0.03, model, ha='center', va='center', 
             fontsize=7, fontweight='bold', color='black')

# Текст Deep Learning
ax1.text(0.96, 0.92, 'Deep Learning\nмодели', ha='right', va='center', 
         fontsize=11, fontweight='bold', color=colors['deep'])

# Ветвь 2: Классическое ML
# Блок: Вычисление признаков
feature_box = FancyBboxPatch((0.61, 0.65), 0.15, 0.1, boxstyle="round,pad=0.03",
                            facecolor=colors['feature'], edgecolor='black', linewidth=2)
ax1.add_patch(feature_box)
feature_text = 'Вычисление\nTD/расширенных\nпризнаков'
ax1.text(0.685, 0.7, feature_text, ha='center', va='center', 
         fontsize=10, fontweight='bold', color='black')

# Стрелка к ML/Feature-based DL
ax1.annotate('', xy=(0.78, 0.7), xytext=(0.61, 0.7),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# ML модели
ml_models = ['SVM\n(RBF)', 'SVM\n(Linear)', 'Random\nForest', 
             'Powerful\nFeatures\n+ MLP', 'XGBoost', 'LDA']
for i, model in enumerate(ml_models):
    x = 0.80 + (i % 3) * 0.06
    y = 0.75 - (i // 3) * 0.08
    model_box = FancyBboxPatch((x, y), 0.055, 0.06, boxstyle="round,pad=0.01",
                              facecolor=colors['classical'], edgecolor='black', linewidth=1.5)
    ax1.add_patch(model_box)
    ax1.text(x + 0.0275, y + 0.03, model, ha='center', va='center', 
             fontsize=7, fontweight='bold', color='black')

# Текст Classical ML
ax1.text(0.96, 0.7, 'Classical ML /\nFeature-based DL', ha='right', va='center', 
         fontsize=11, fontweight='bold', color=colors['classical'])

# Добавляем подписи потоков
ax1.text(0.48, 0.92, 'Поток 1: Глубокое обучение', fontsize=10, fontweight='bold', 
         color=colors['deep'], ha='center')
ax1.text(0.48, 0.68, 'Поток 2: Признаковый подход', fontsize=10, fontweight='bold', 
         color=colors['classical'], ha='center')

# ==================== ПАНЕЛЬ 2: Схема LOSO ====================

# Множество субъектов S = {1,...,40}
ax2.text(0.1, 0.9, 'Множество субъектов: S = {1, 2, ..., 40}', 
         fontsize=12, fontweight='bold', ha='left')

# Рисуем субъектов (первые 8 + ... + последние 8 для наглядности)
subjects = list(range(1, 9)) + ['...'] + list(range(33, 41))
for i, subj in enumerate(subjects):
    x = 0.05 + i * 0.05
    if subj == '...':
        ax2.text(x, 0.75, '...', fontsize=14, ha='center', va='center')
    else:
        subject_circle = Circle((x, 0.75), 0.02, facecolor=colors['subject'], 
                               edgecolor='black', linewidth=2)
        ax2.add_patch(subject_circle)
        ax2.text(x, 0.75, str(subj), fontsize=9, fontweight='bold', 
                ha='center', va='center')

# Показываем один шаг LOSO
# Выбираем тестовый субъект (например, 9)
test_subject = 9
test_idx = subjects.index(test_subject) if test_subject in subjects else 4  # fallback

# Подпись шага
ax2.text(0.5, 0.6, 'Для каждого k = 1,...,40:', fontsize=11, fontweight='bold', 
         ha='center', va='center')

# Блок Train
train_box = FancyBboxPatch((0.15, 0.4), 0.3, 0.2, boxstyle="round,pad=0.05",
                          facecolor=colors['train'], edgecolor='black', linewidth=2, alpha=0.8)
ax2.add_patch(train_box)
ax2.text(0.3, 0.5, 'Train:\nS \\ {k}\n(39 субъектов)', ha='center', va='center', 
         fontsize=11, fontweight='bold', color='black')

# Блок внутри Train для Validation
val_box = FancyBboxPatch((0.18, 0.42), 0.24, 0.08, boxstyle="round,pad=0.02",
                        facecolor=colors['val'], edgecolor='black', linewidth=1.5)
ax2.add_patch(val_box)
ax2.text(0.3, 0.46, 'Validation:\nподмножество train subjects', ha='center', va='center', 
         fontsize=9, fontweight='bold', color='black')

# Блок Test
test_box = FancyBboxPatch((0.55, 0.4), 0.3, 0.2, boxstyle="round,pad=0.05",
                         facecolor=colors['test'], edgecolor='black', linewidth=2, alpha=0.8)
ax2.add_patch(test_box)
ax2.text(0.7, 0.5, 'Test:\nsubject k', ha='center', va='center', 
         fontsize=11, fontweight='bold', color='black')

# Соединяющие стрелки
# От субъектов к Train и Test
for i, subj in enumerate(subjects):
    if subj == '...':
        continue
    x = 0.05 + i * 0.05
    # Все кроме тестового идут в Train
    if subj != test_subject:
        ax2.annotate('', xy=(0.15, 0.55), xytext=(x, 0.73),
                    arrowprops=dict(arrowstyle='->', lw=1, color='gray', alpha=0.6))
    # Тестовый идет в Test
    else:
        ax2.annotate('', xy=(0.55, 0.55), xytext=(x, 0.73),
                    arrowprops=dict(arrowstyle='->', lw=2, color='red'))

# Подсветка тестового субъекта
if test_subject in subjects:
    test_x = 0.05 + subjects.index(test_subject) * 0.05
    highlight_circle = Circle((test_x, 0.75), 0.025, facecolor='none', 
                             edgecolor='red', linewidth=3, linestyle='--')
    ax2.add_patch(highlight_circle)
    ax2.text(test_x, 0.82, f'k = {test_subject}', fontsize=10, fontweight='bold', 
             color='red', ha='center')

# Блок усреднения метрик
metrics_box = FancyBboxPatch((0.3, 0.15), 0.4, 0.15, boxstyle="round,pad=0.05",
                            facecolor='#F0F0F0', edgecolor='black', linewidth=2)
ax2.add_patch(metrics_box)
metrics_text = 'Усреднение метрик по всем k:\n• Accuracy\n• F1-score\n• Precision/Recall\n• Confusion Matrix'
ax2.text(0.5, 0.225, metrics_text, ha='center', va='center', 
         fontsize=10, fontweight='bold', color='black')

# Стрелки от Train и Test к метрикам
ax2.annotate('', xy=(0.3, 0.3), xytext=(0.3, 0.4),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax2.annotate('', xy=(0.7, 0.3), xytext=(0.7, 0.4),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Общая подпись процесса LOSO
ax2.text(0.5, 0.07, 'Процесс повторяется для каждого субъекта k = 1,...,40', 
         fontsize=11, fontstyle='italic', ha='center')

# Настройка отступов и сохранение
plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.1)

# Сохраняем в высоком разрешении
plt.savefig('pipeline_loso_illustration.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.savefig('pipeline_loso_illustration.pdf', bbox_inches='tight', pad_inches=0.1)

print("Иллюстрация сохранена как 'pipeline_loso_illustration.png' и 'pipeline_loso_illustration.pdf'")
plt.show()