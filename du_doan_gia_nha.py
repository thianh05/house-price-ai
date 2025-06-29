import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dữ liệu
data = {
    'DienTich': [30, 50, 70, 90, 110],
    'Gia': [500, 700, 900, 1100, 1300]
}
df = pd.DataFrame(data)

# Tách input/output
x = df[['DienTich']]
y = df['Gia']

# Mô hình
model = LinearRegression()
model.fit(x, y)

# Dự đoán (dùng DataFrame để tránh cảnh báo)
dien_tich_moi = pd.DataFrame({'DienTich': [80]})
gia_du_doan = model.predict(dien_tich_moi)

# In kết quả đơn giản để tránh lỗi font
print("Gia du doan cho nha", dien_tich_moi.iloc[0, 0], "m2 la:", round(gia_du_doan[0]), "trieu dong")

# Vẽ biểu đồ
plt.scatter(x, y, color='blue', label='Dữ liệu thực tế')
plt.plot(x, model.predict(x), color='red', label='Đường hồi quy')
plt.xlabel('Diện tích (m²)')
plt.ylabel('Giá (triệu đồng)')
plt.title('Dự đoán giá nhà theo diện tích')
plt.legend()
plt.show()
