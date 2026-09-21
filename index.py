import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
random_numbers = np.random.randint(-10000, 10001, size=1000)
data_series = pd.Series(random_numbers)

print("Первые 10 сгенерированных значений:")
print(data_series.head(10))
print(f"\nРазмер набора данных: {len(data_series)} элементов")

data_series.to_csv('generated_data.csv', index=False, header=['values'])
loaded_series = pd.read_csv('generated_data.csv')
loaded_series = loaded_series['values']

missing_values = data_series.isnull().sum()
print(f"Количество пропущенных значений: {missing_values}")

rounded_for_hist = data_series.apply(lambda x: round(x / 100) * 100)
print("\nПример округления первых 5 значений:")
print("Исходные:", data_series.head(5).values)
print("Округленные:", rounded_for_hist.head(5).values)

np.random.seed(123)
data_list = np.random.randint(-10000, 10001, 1000)
original_series = pd.Series(data_list, name='Исходные данные')

print("\nDataset успешно создан.")
print(f"Тип объекта: {type(original_series)}")
print(f"Количество элементов: {original_series.size}")
print(original_series.describe())

min_value = original_series.min()
print(f"1. Минимальное значение: {min_value}")

unique_count = original_series.nunique()
duplicates_count = len(original_series) - unique_count
print(f"2. Количество повторяющихся элементов: {duplicates_count}")
print(f"   Всего уникальных значений: {unique_count} из {len(original_series)}")

max_value = original_series.max()
print(f"3. Максимальное значение: {max_value}")

sum_values = original_series.sum()
print(f"4. Сумма всех значений: {sum_values}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(original_series, color='blue', linewidth=0.5)
plt.title('Линейный график исходных данных')
plt.xlabel('Индекс элемента')
plt.ylabel('Значение')
plt.grid(True, linestyle='--', alpha=0.6)

rounded_values = original_series.apply(lambda x: round(x / 100) * 100)
plt.subplot(1, 2, 2)
plt.hist(rounded_values, bins=40, color='green', alpha=0.7, edgecolor='black')
plt.title('Гистограмма данных, округленных до сотен')
plt.xlabel('Диапазоны значений (сотни)')
plt.ylabel('Частота')
plt.grid(True, linestyle='--', alpha=0.6, axis='y')
plt.tight_layout()
plt.show()

analysis_df = pd.DataFrame()
analysis_df['Исходные'] = original_series
analysis_df['По возрастанию'] = original_series.sort_values().reset_index(drop=True)
analysis_df['По убыванию'] = original_series.sort_values(ascending=False).reset_index(drop=True)

print("\nПервые 10 строк полученного DataFrame:")
print(analysis_df.head(10))
print(f"\nРазмер DataFrame: {analysis_df.shape}")

plt.figure(figsize=(12, 6))
plt.plot(analysis_df['По возрастанию'], label='По возрастанию', color='red', linewidth=1.5)
plt.plot(analysis_df['По убыванию'], label='По убыванию', color='blue', linewidth=1.5)
plt.title('Сравнение отсортированных рядов данных')
plt.xlabel('Номер элемента в отсортированном порядке')
plt.ylabel('Значение')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.fill_between(range(len(analysis_df)), analysis_df['По возрастанию'], analysis_df['По убыванию'], color='gray', alpha=0.1)
plt.show()

plt.figure(figsize=(10, 6))
plt.boxplot(original_series, vert=False, patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red', linewidth=2),
            whiskerprops=dict(color='blue'),
            capprops=dict(color='blue'),
            flierprops=dict(marker='o', markerfacecolor='red',
                            markersize=4, alpha=0.5))
plt.title('Ящик с усами (boxplot) исходных данных')
plt.xlabel('Значение')
plt.grid(True, linestyle='--', alpha=0.6, axis='x')
plt.tight_layout()
plt.show()