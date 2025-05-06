import pyreadstat
import pandas as pd

data_path = r"D:\working\老年预测\A017-CLHLS中国老年人健康长寿影响因素调查\A017-CLHLS中国老年人健康长寿影响因素调查\CLHLS_2018_cross_sectional_dataset_15874\clhls_2018_cross_sectional_dataset_15874.sav"

# 指定cp1254编码读取文件
df, meta = pyreadstat.read_sav(data_path, encoding='cp1254')

# 将数据保存为Excel文件
excel_path = r"D:\working\老年预测\clhls_2018_cross_sectional_dataset_15874.xlsx"
df.to_excel(excel_path, index=False)
print(f"数据已成功保存为 {excel_path}")