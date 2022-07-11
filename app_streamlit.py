import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# titúlo
st.write('''Prevendo Diabetes\n
Ap que utiliza machine learning para preve possível diabetes dos pacientes.\n
Fonte: Pima - India(Kaggle)''')

# dataset
df = pd.read_csv('C:/diabetes.csv')

# cabeçalho
st.subheader('Informações dos Dados')

# nome do usuário
user_input = st.sidebar.text_input('Diga seu nome: ')

st.write(f'Paciente: {user_input}')
# dados de entrada
x = df.drop(['Outcome'], 1)
y = df['Outcome']

# separa daasdos treinamento e teste
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=2, random_state=42)


# dados do usuário com a função
def get_user_data():
    pregnancies = st.sidebar.slider('Gravidez', 0, 15, 1)
    glucose = st.sidebar.slider('Glicose', 0, 200, 110)
    blood_pressure = st.sidebar.slider('Pressão Sanguínea', 0, 122, 72)
    skin_thickness = st.sidebar.slider('Espessura da pele', 0, 99, 20)
    insulin = st.sidebar.slider('Insulina', 0, 900, 38)
    bmi = st.sidebar.slider('Indice de massa corporal', 0.0, 70.0, 15.0)
    dpf = st.sidebar.slider('Histórico familiar de diabetes', 0.0, 3.0, 0.0)
    age = st.sidebar.slider('idade', 15, 100, 21)

    user_data = {'Gravidez': pregnancies,
                 'Glicose': glucose,
                 'Pressao sanguinea': blood_pressure,
                 'Espessura da pele': skin_thickness,
                 'Insulina': insulin,
                 'Indice de massa corporal': bmi,
                 'Historico familiar de diabetes': dpf,
                 'Idade': age }
    features =pd.DataFrame(user_data, index=[0])

    return features

user_input_variables = get_user_data()

# grafico
graf = st.bar_chart(user_input_variables)

st.subheader('Dados do usuário')
st.write(user_input_variables)

dtc =DecisionTreeClassifier(criterion='entropy', max_depth=3)
dtc.fit(X_train, Y_train)

# acuracia do modelo
st.subheader('Acurácia do modelo')
st.write(accuracy_score(Y_test, dtc.predict(X_test)) * 100)

# previsão
prediction = dtc.predict((user_input_variables))

st.subheader('Previsão')
st.write(prediction)