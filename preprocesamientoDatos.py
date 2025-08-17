import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import streamlit as st
#from sklearn.tree import export_graphviz
#mport graphviz



dataset = pd.read_csv("D:\Propietario\Desktop\TODO\Modulares\ENFERMEDADES PROYECTO\DATASET\datosLimpios.csv")
df = pd.DataFrame(dataset) #generar el dataframe para su manipulacion

df = df.sample(frac=1).reset_index(drop=True)

X = df.drop(columns=['riesgo_hipertension'])
y = df['riesgo_hipertension'] #variable objetivo

#Dividir el conjunto de prueba (20% del total de los datos)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#uso del modelo
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

#evaluar el modelo
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

f1_score = f1_score(y_test, y_pred)
print("F1 Score:", f1_score)

confusion_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de confusión:")
print(confusion_matrix)



# Visualizar el árbol de decisión con Graphviz para mejor calidad
#dot_data = export_graphviz(model, out_file=None, feature_names=X.columns, filled=True, rounded=True, special_characters=True) #class_names=['No', 'Si']
#graph = graphviz.Source(dot_data)
#graph
#graph.render("arbol_decision")

    #solicitud de datos

#entrada de datos
sexo_entrada = st.number_input("Sexo (0 para mujer, 1 para hombre):")
edad_entrada = st.number_input("Edad:")
peso_entrada = st.number_input("Peso:", value=0)
estatura_entrada = st.number_input("Estatura en cm" , value=0)
#resultado_glucosa_entrada = st.number_input("Resultado de glucosa:")
tension_arterial_entrada = st.number_input("Tensión arterial:")

# Validamos que la estatura no sea cero para evitar la división por cero.
if estatura_entrada > 0:
    # La estatura debe estar en metros para la fórmula del IMC.
    estatura_metros = estatura_entrada / 100
    masa_corporal = peso_entrada / (estatura_metros ** 2)
    st.success(f"Tu Índice de Masa Corporal (IMC) es: {masa_corporal:.2f}")
else:
    # Muestra un mensaje de error o una indicación al usuario.
    st.warning("Ingresa un valor de estatura mayor a 0 para calcular el IMC.")


if st.button("Predecir"):
    #input_data = [[sexo_entrada, edad_entrada, resultado_glucosa_entrada, estatura_entrada, peso_entrada, tension_arterial_entrada, masa_corporal]]
    input_data = [[sexo_entrada, edad_entrada, estatura_entrada, peso_entrada, masa_corporal, tension_arterial_entrada]]
    input_data = pd.DataFrame(input_data, columns=X.columns)

    prediccion = model.predict(input_data)

    if prediccion[0] == 0:
        st.success("El paciente no tiene riesgo de hipertension.")
    else:
        st.warning("El paciente tiene riesgo de hipertension.")


st.title("Modelo de Árboles de Decisión")
st.write("Accuracy:", accuracy)
st.write("F1 Score:", f1_score)
st.write("Matriz de confusión:")
st.write(confusion_matrix)

