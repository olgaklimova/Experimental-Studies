import sys
import pandas as pd
import pickle
import category_encoders as ce

## test_df не должен содержать строки с неизвестными значениями и строки при которых  'State' != 'COMPLETED' ##
class SKPredModel:
    def __init__(self, model_path):
        #Инициализация модели
        with open(model_path, 'rb') as f:
          self.model = pickle.load(f)
        
    def prepare_df(self, test_df):
        src_index = test_df.index
        data = test_df
        #Проверяем, чтобы не было строк с неизвестными значениями
        assert not data.isnull().values.any(), "Обнаружены строки с неизвестными значениями, пожалуйста заполните их"
        #Проверяем, чтобы все задачи в тестовых данных были успешно завершены
        assert (data['State'] == 'COMPLETED').all(), "Обнаружены значения, отличные от 'COMPLETED' в столбце 'State', пожалуйста исправьте это"
        
        #Удаление колонки 'State'
        data = data.drop(columns=['State'])
        
        #Преобразование категориальных признаков
        #One-Hot Encoding
        #признак "Area"
        unique_area = data['Area'].unique()
        #Обработка случая, когда нет одной area из 'astrophys' 'phys' 'mech' 'energ' 'mach' 'radiophys' 'biophys' 'geophys' 'bioinf' 'it'
        area_arr = ['astrophys', 'phys', 'mech', 'energ', 'mach', 'radiophys', 'biophys', 'geophys', 'bioinf', 'it']
        n = len(area_arr)
        if len(unique_area) != n:
           #Проверяем, каких колонок нет из 'astrophys' 'phys' 'mech' 'energ' 'mach' 'radiophys' 'biophys' 'geophys' 'bioinf' 'it'
           missing_area = [i for i in area_arr if i not in unique_area]
           data = data.reindex(columns = data.columns.tolist() + missing_area, fill_value=0)
        one_hot_encoding_1_data = pd.get_dummies(data['Area'], dtype=int)
        #добавление новых колонок к исходному датафрейму
        data = pd.concat([data, one_hot_encoding_1_data], axis=1)
        data = data.drop(columns=['Area'])
        #признак "Partition"
        unique_partition = data['Partition'].unique()
        #Обработка случая, когда нет одного partition из 'cascade', 'g2', 'nv', 'tornado', 'tornado-k40'
        partition_arr = ['cascade', 'g2', 'nv', 'tornado', 'tornado-k40']
        n = len(partition_arr)
        if len(unique_partition) != n:
           #Проверяем, каких колонок нет из 'cascade', 'g2', 'nv', 'tornado', 'tornado-k40'
           missing_partition = [i for i in partition_arr if i not in unique_partition]
           data = data.reindex(columns = data.columns.tolist() + missing_partition, fill_value=0)
        one_hot_encoding_2_data = pd.get_dummies(data['Partition'], dtype=int)
        #добавление новых колонок к исходному датафрейму
        data = pd.concat([data, one_hot_encoding_2_data], axis=1)
        data = data.drop(columns=['Partition'])
    
        #Применение parse_time к каждой строке столбца
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
        
        #Удаление ExitCode
        data = data.drop(columns=['ExitCode'])
         
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
        return data.loc[src_index]
     
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
        for key in self.model.feature_name():
          assert key in prepared_data.columns, f"{key} column missed in test_df"
          
        #Предсказание модели
        Y_pred = self.model.predict(prepared_data)
        return pd.Series(Y_pred)

if __name__ == '__main__':
    #Проверка созданного класса
    model_path = 'model_1.pkl'
    #model_path = 'model_2.pkl'
    model = SKPredModel(model_path)
    
    test_df = pd.read_csv('C:\\Users\\Unicorn\\Desktop\\SK_LGBM_Klimova\\train_w_areas_st_till_june.csv', index_col=0)
    #Удаляем строки с неизвестными значениями и строки при которых  'State' != 'COMPLETED'
    test_df = test_df.dropna()
    test_df = test_df.loc[test_df['State'] == 'COMPLETED']
    prepared_data = model.prepare_df(test_df)
    
    Y_pred = model.predict(prepared_data)