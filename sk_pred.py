import pandas as pd
import pickle
import category_encoders as ce
from sklearn.metrics import r2_score


class SKPredModel:
    def __init__(self):
        
        #Инициализация модели №1
        with open('model_1.pkl', 'rb') as f:
          self.model = pickle.load(f)
            
        #Инициализация модели №2
        #with open('model_2.pkl', 'rb') as f:
        # self.model = pickle.load(f)
        
    def prepare_df(self, test_df):
        
        data = test_df
        #Удаление строк с неизвестными значениями
        data = data.dropna()
        #Удаление колонки с индексами строк
        first_column_name = data.columns[0]
        data = data.drop(columns=[first_column_name])
        
        #Преобразование категориальных признаков
        #One-Hot Encoding
        #признак "Area"
        one_hot_encoding_1_data = pd.get_dummies(data['Area'])
        #добавление новых колонок к исходному датафрейму
        data = pd.concat([data, one_hot_encoding_1_data], axis=1)
        data = data.drop(columns=['Area'])
        #признак "Partition"
        one_hot_encoding_2_data = pd.get_dummies(data['Partition'])
        #добавление новых колонок к исходному датафрейму
        data = pd.concat([data, one_hot_encoding_2_data], axis=1)
        data = data.drop(columns=['Partition'])
    

        # Применение parse_time к каждой строке столбца
        data['Elapsed_Seconds'] = data['Elapsed'].apply(self.parse_time)
        data = data.drop(columns=['Elapsed'])
        data['Timelimit_Seconds'] = data['Timelimit'].apply(self.parse_time)
        data = data.drop(columns=['Timelimit'])
        
        #Обработка еще одного категориального признака
        encoder = ce.TargetEncoder()
        target_encoding_1_data = encoder.fit_transform(data['JobName'], data['Elapsed_Seconds'])
        #Добавление новой колонки
        data = pd.concat([data, target_encoding_1_data], axis=1)
        data = data.drop(columns=['JobName'])
        
        #Создание нового признака TimeWeit
        data['Submit'] = pd.to_datetime(data['Submit'])
        data['Start'] = pd.to_datetime(data['Start'])
        data['TimeWeit'] = data['Start'] - data['Submit']
        data = data.drop(columns=['Start'])
        data = data.drop(columns=['Submit'])
        
        #Преобразование данных TimeWeit
        data['TimeWeit_Seconds'] = data['TimeWeit'].apply(self.parse_time_new)
        data = data.drop(columns=['TimeWeit'])
        
        #Удаляем ExitCode
        data = data.drop(columns=['ExitCode'])
        #Удаляем все неудачно завершенные задачи
        data = data.loc[data['State'] == 'COMPLETED']
        data = data.drop(columns=['State'])
        
        #Сортировка по UID
        data = data.sort_values(by='UID')
        
        #Создание столбца Mean_Elapsed_Seconds со средним значением Elapsed_Seconds по UID
        data['Mean_Elapsed_Seconds'] = data.groupby('UID')['Elapsed_Seconds'].transform('mean')
        
        #Создание столбцов с различными размерами окон средних
        #размеры окон
        window_sizes = [2, 4, 8, 16, 32, 64]
        for window_size in window_sizes: 
            #рассчитываем скользящее среднее
            rolling_mean = data.groupby('UID')['Elapsed_Seconds'].rolling(window=window_size, min_periods=1).mean()
            #преобразуем rolling_mean в DataFrame
            rolling_mean_df = rolling_mean.reset_index(level=0, drop=True)
            #добавляем столбец со скользящим средним в исходный DataFrame
            data[f'Mean_{window_size}_Elapsed_Seconds'] = rolling_mean_df
        
        #Удаление значений Y
        data = data.drop(columns=['Elapsed_Seconds'])
        return data
    
    def parse_time(self, time_str):
        #Преобразование форматов столцов Elapsed_Seconds и Timelimit_Seconds
        if '-' in time_str:
          days, time = time_str.split('-')
        else:
          days = '0'
          time = time_str
        hours, minutes, seconds = map(int, time.split(':'))
        result = int(days)*86400 + hours*3600 + minutes*60 + seconds
        return result
    
    def parse_time_new(self, time_str):
        #разделение строки времени на дни и время
        days, time = str(time_str).split(' days ')
        hours, minutes, seconds = map(int, time.split(':'))
        result = int(days)*86400 + hours*3600 + minutes*60 + seconds
        return result
    
    def predict(self, prepared_data):
        #Предсказание модели
        Y_pred = self.model.predict(prepared_data)
        return pd.Series(Y_pred)

#Проверка созданного класса
model = SKPredModel()
test_df = pd.read_csv('C:\\Users\\Unicorn\\Desktop\\SK_LGBM_Klimova\\train_w_areas_st_till_june.csv')
prepared_data = model.prepare_df(test_df)
Y_pred = model.predict(prepared_data)