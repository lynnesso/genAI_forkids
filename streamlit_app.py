import streamlit as st
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

st.title("🎓 Рекомендательная система детских секций")

# Пользовательский ввод
age = st.slider("Возраст ребёнка", 5, 17, 9)
budget = st.slider("Бюджет на секции (руб/мес)", 0, 20000, 5000)
interests = st.selectbox("Основной интерес", ["спорт", "творчество", "логика", "искусство", "программирование"])
direction = st.selectbox("Направление развития", ["спорт", "искусство", "STEM"])
district = st.selectbox("Район проживания", ["Центральный", "Южный", "Северный", "Западный", "Восточный"])

# Заглушка для обучения
data = pd.DataFrame({
    "age": [7, 9, 10, 8, 6, 11],
    "interests": ["спорт", "творчество", "логика", "спорт", "искусство", "спорт"],
    "direction": ["спорт", "искусство", "STEM", "спорт", "искусство", "STEM"],
    "budget": [3000, 4000, 6000, 5000, 2500, 7000],
    "district": ["Центральный", "Южный", "Западный", "Центральный", "Восточный", "Северный"],
    "target_section": ["айкидо", "рисование", "робототехника", "баскетбол", "лепка", "шахматы"]
})

cat_features = data[["interests", "direction", "district"]]
num_features = data[["age", "budget"]]
target = data["target_section"]

enc = OneHotEncoder()
scaled = StandardScaler()
target_enc = OneHotEncoder()

X_cat = enc.fit_transform(cat_features).toarray()
X_num = scaled.fit_transform(num_features)
X = np.concatenate([X_num, X_cat], axis=1)

target_enc.fit(target.values.reshape(-1, 1))
y = target_enc.transform(target.values.reshape(-1, 1)).toarray()

# Ввод пользователя
user_df = pd.DataFrame([[age, budget, interests, direction, district]],
                       columns=["age", "budget", "interests", "direction", "district"])
user_num = scaled.transform(user_df[["age", "budget"]])
user_cat = enc.transform(user_df[["interests", "direction", "district"]]).toarray()
user_input = torch.tensor(np.concatenate([user_num, user_cat], axis=1), dtype=torch.float32)

# Модель генератора
class Generator(nn.Module):
    def __init__(self, noise_dim, cond_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=1)
        )
    def forward(self, noise, cond):
        x = torch.cat([noise, cond], dim=1)
        return self.model(x)

cond_dim = X.shape[1]
output_dim = y.shape[1]
noise_dim = 10
G = Generator(noise_dim, cond_dim, output_dim)
# ⚠️ Здесь можно загрузить обученные веса (если есть)

# Генерация рекомендации
if st.button("Сгенерировать рекомендации"):
    noise = torch.randn(1, noise_dim)
    with torch.no_grad():
        output = G(noise, user_input)
    recommendation = target_enc.inverse_transform(output.detach().numpy())[0][0]
    st.success(f"✅ Рекомендуемая секция: **{recommendation}**")
