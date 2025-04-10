import base64
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Configuração da página com tema escuro
st.set_page_config(
    page_title="NeuroAnalytics - Dashboard de AVC",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paleta de cores personalizada
COLOR_PALETTE = {
    'background': '#0E1117',
    'card': '#1E2130',
    'text': '#FFFFFF',
    'primary': '#4B8BBE',
    'secondary': '#306998',
    'accent': '#FFE873',
    'danger': '#FF6B6B',
    'success': '#6BCB77'
}

# CSS
st.markdown(f"""
<style>
    * {{ outline: none !important; box-shadow: none !important; }}
    .stApp {{ background-color: {COLOR_PALETTE['background']}; color: {COLOR_PALETTE['text']}; }}
    .stMetric {{ background-color: {COLOR_PALETTE['card']}; border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3); height: 120px; }}
    .fixed-header {{ position: fixed; top: 0; background-color: {COLOR_PALETTE['card']}; border-bottom: 1px solid {COLOR_PALETTE['secondary']}; }}
    .info-card {{ background-color: {COLOR_PALETTE['card']}; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
    .download-btn {{ background: none !important; border: none !important; color: {COLOR_PALETTE['primary']} !important; }}
    .download-btn:hover {{ text-decoration: underline !important; color: {COLOR_PALETTE['secondary']} !important; }}
    div[data-testid="stForm"], div[data-testid="stVerticalBlock"], 
    div[data-testid="stHorizontalBlock"], div[data-testid="stExpander"] {{ border: none !important; }}
    .stTextInput input:focus, .stNumberInput input:focus, .stSelectbox select:focus,
    .stSlider div:focus, .stRadio div:focus, .stCheckbox div:focus {{
        box-shadow: none !important; outline: none !important; border-color: {COLOR_PALETTE['secondary']} !important;
    }}
</style>
""", unsafe_allow_html=True)

# JavaScript atualizado para comunicação com Streamlit
st.markdown("""
<script>
function setPage(page) {
    Streamlit.setComponentValue(page);
}
</script>
""", unsafe_allow_html=True)

# Trecho para lidar com a navegação
if 'page' not in st.session_state:
    st.session_state.page = 'Painel Geral'

# Atualiza o estado da página quando um botão é clicado
def update_page():
    st.session_state.page = st.session_state.selected_page

# Componente invisível para comunicação com o JavaScript
page = st.selectbox(
    'Page selector',
    options=['Painel Geral', 'Modelo Preditivo', 'Sobre o Projeto'],
    key='selected_page',
    on_change=update_page,
    label_visibility='collapsed'
)

# CSS personalizado
st.markdown(f"""
<style>
    .stApp {{
        background-color: {COLOR_PALETTE['background']};
        color: {COLOR_PALETTE['text']};
    }}
    .css-1aumxhk {{
        background-color: {COLOR_PALETTE['card']};
    }}
    .st-bb {{
        background-color: transparent;
    }}
    .st-at {{
        background-color: {COLOR_PALETTE['card']};
    }}
    .st-ae {{
        background-color: {COLOR_PALETTE['card']};
    }}
    .stMetric {{
        background-color: {COLOR_PALETTE['card']};
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}
    .metric-value {{
        font-size: 28px !important;
        font-weight: bold !important;
    }}
    .metric-label {{
        font-size: 14px !important;
    }}
    .css-1v3fvcr {{
        color: {COLOR_PALETTE['text']};
    }}
    .css-qri22k {{
        background-color: {COLOR_PALETTE['card']};
    }}
    .css-1y4p8pa {{
        max-width: 100%;
        padding: 1rem;
    }}
    .stButton>button {{
        background-color: {COLOR_PALETTE['primary']};
        color: white;
        border-radius: 5px;
        transition: all 0.3s ease;
    }}
    .stButton>button:hover {{
        background-color: {COLOR_PALETTE['secondary']};
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }}
    
    /* Navegação sofisticada */
    .nav-container {{
        display: flex;
        justify-content: center;
        margin-bottom: 30px;
    }}
    .nav-item {{
        padding: 12px 24px;
        margin: 0 5px;
        border-radius: 30px;
        cursor: pointer;
        color: white;
        background-color: {COLOR_PALETTE['card']};
        transition: all 0.3s ease;
        font-weight: normal;
        border: 1px solid {COLOR_PALETTE['secondary']};
    }}
    .nav-item:hover {{
        background-color: {COLOR_PALETTE['secondary']};
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }}
    .nav-item.active {{
        background-color: {COLOR_PALETTE['primary']};
        font-weight: bold;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }}
    
    /* Cards com altura igual */
    .stMetric {{
        min-height: 120px;
    }}
</style>
""", unsafe_allow_html=True)

# Carregar dados
@st.cache_data
def load_data():
    return pd.read_csv('base_filtrada_limpa.csv')

df = load_data()

# Mapeamento de códigos para rótulos
mapeamento = {
    'C006': {1: 'Homem', 2: 'Mulher'},
    'C009': {1: 'Branca', 2: 'Preta', 3: 'Amarela', 4: 'Parda', 5: 'Indígena', 9: 'Ignorado'},
    'I00102': {1: 'Sim', 2: 'Não', 9: 'Ignorado'},
    'P02601': {1: 'Muito alto', 2: 'Alto', 3: 'Adequado', 4: 'Baixo', 5: 'Muito baixo', 9: 'Ignorado'},
    'P027': {1: 'Nunca', 2: 'Menos de 1x/mês', 3: '1x ou mais/mês', 9: 'Ignorado'},
    'P034': {1: 'Sim', 2: 'Não', 9: 'Ignorado'},
    'P050': {1: 'Sim', 2: 'Não', 9: 'Ignorado'},
    'Q00201': {1: 'Sim', 2: 'Não', 9: 'Ignorado'},
    'Q06306': {1: 'Sim', 2: 'Não', 9: 'Ignorado'},
    'Q068': {1: 'Sim', 2: 'Não', 9: 'Ignorado'},
    'Q092': {1: 'Sim', 2: 'Não', 9: 'Ignorado'},
    'Q11006': {1: 'Sim', 2: 'Não', 9: 'Ignorado'}
}

# Pré-processamento para o modelo corrigido
def preprocess_data(df):
    # Filtrar apenas colunas relevantes
    features = ['C006', 'C008', 'C009', 'I00102', 'P00104', 'P00404', 
                'P00901', 'P015', 'P018', 'P02002', 'P02501', 'P02602',
                'P02601', 'P027', 'P034', 'P042', 'P044', 'P04501', 
                'P04502', 'P050', 'Q00201', 'Q06306', 'Q092', 'Q11006']
    
    df_model = df[features + ['Q068']].copy()
    
    # Remover linhas com valores ignorados (9)
    for col in df_model.columns:
        if col in mapeamento:
            df_model = df_model[df_model[col] != 9]
    
    # Converter variáveis categóricas
    for col in df_model.columns:
        if col in mapeamento:
            df_model[col] = df_model[col].map(mapeamento[col])
    
    # One-hot encoding
    df_model = pd.get_dummies(df_model, drop_first=True)
    
    return df_model

df_model = preprocess_data(df)

# Treinar modelo com balanceamento de classes
@st.cache_resource
def train_model():
    X = df_model.drop('Q068_Sim', axis=1)
    y = df_model['Q068_Sim']
    
    # Balanceamento de classes
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(n_estimators=150, 
                                 max_depth=10,
                                 min_samples_split=5,
                                 class_weight='balanced',
                                 random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    return model, accuracy, roc_auc, X.columns

model, accuracy, roc_auc, feature_names = train_model()

# Estado da sessão para controlar a página ativa
if 'page' not in st.session_state:
    st.session_state.page = 'Painel Geral'


# Página 1: Painel Geral
if st.session_state.page == 'Painel Geral':
    st.title("Análise de Fatores de Risco para AVC")
    
    def format_number(num):
        return f"{num:,.0f}".replace(",", ".")
    
    # Cards com métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_casos = len(df)
        st.metric("Total de Casos Analisados", format_number(total_casos))
    
    with col2:
        avc_casos = df['Q068'].value_counts().get(1, 0)
        st.metric("Casos de AVC", 
                 format_number(avc_casos), 
                 f"{(avc_casos/total_casos*100):.1f}%")
    
    with col3:
        media_idade = df['C008'].mean()
        st.metric("Idade Média", f"{media_idade:.1f} anos")
    
    with col4:
        hipertensos = df['Q00201'].value_counts().get(1, 0)
        st.metric("Hipertensos", 
                 format_number(hipertensos), 
                 f"{(hipertensos/total_casos*100):.1f}%")
        st.caption("*Uma das principais comorbidades associada ao AVC")
    
    # Gráfico 1: Distribuição de AVC por sexo e idade (apenas casos com AVC)
    st.subheader("Distribuição de AVC por Sexo e Idade")
    df_avc = df[df['Q068'] == 1].copy()
    df_avc['C006'] = df_avc['C006'].map({1: 'Homem', 2: 'Mulher'})
    
    fig1 = px.histogram(df_avc, x='C008', color='C006',
                       labels={'C008': 'Idade', 'C006': 'Sexo'},
                       color_discrete_map={'Homem': COLOR_PALETTE['primary'], 
                                         'Mulher': COLOR_PALETTE['danger']},
                       nbins=20,
                       template='plotly_dark')
    fig1.update_layout(barmode='stack', plot_bgcolor=COLOR_PALETTE['card'],
                     paper_bgcolor=COLOR_PALETTE['card'])
    st.plotly_chart(fig1, use_container_width=True)
    
    # Gráfico 2: Fatores de risco principais (apenas casos com AVC)
    st.subheader("Prevalência de Fatores de Risco em Pacientes com AVC")
    fatores = ['Q00201', 'Q06306', 'P050', 'P02601', 'P027']
    fatores_labels = ['Hipertensão', 'Doença Cardíaca', 'Tabagismo', 'Consumo de Sal Alto', 'Consumo de Álcool']
    
    def calculate_prevalence(col):
        if col == 'P02601':
            return (df_avc[col].isin([1, 2])).mean() * 100  # Muito alto ou alto
        elif col == 'P027':
            return (df_avc[col] == 3).mean() * 100  # 1x ou mais por mês
        else:
            return (df_avc[col] == 1).mean() * 100
    
    prevalencia = [calculate_prevalence(factor) for factor in fatores]
    
    fig2 = px.bar(x=fatores_labels, y=prevalencia, 
                 labels={'x': 'Fator de Risco', 'y': 'Prevalência (%)'},
                 color=fatores_labels,
                 color_discrete_sequence=[COLOR_PALETTE['primary'], 
                                        COLOR_PALETTE['secondary'],
                                        COLOR_PALETTE['danger'],
                                        COLOR_PALETTE['accent'],
                                        COLOR_PALETTE['success']],
                 template='plotly_dark')
    fig2.update_layout(plot_bgcolor=COLOR_PALETTE['card'],
                     paper_bgcolor=COLOR_PALETTE['card'],
                     showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)
    
    # Gráfico 3: Distribuição por raça/cor (apenas casos com AVC)
    st.subheader("Distribuição de AVC por Raça/Cor")
    df_avc['C009'] = df_avc['C009'].map(mapeamento['C009'])
    
    fig3 = px.pie(df_avc, names='C009', color='C009',
                 labels={'C009': 'Raça/Cor'},
                 color_discrete_sequence=px.colors.qualitative.Dark2,
                 template='plotly_dark')
    fig3.update_layout(plot_bgcolor=COLOR_PALETTE['card'],
                     paper_bgcolor=COLOR_PALETTE['card'])
    st.plotly_chart(fig3, use_container_width=True)

# Página 2: Modelo Preditivo
elif st.session_state.page == 'Modelo Preditivo':
    st.title("Modelo Preditivo de Risco de AVC")
    
    # Cards de desempenho
    st.subheader("Desempenho do Modelo")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="📊 Acurácia do Modelo",
            value=f"{accuracy*100:.1f}%",
            help="Porcentagem de previsões corretas"
        )
    
    with col2:
        st.metric(
            label="📈 AUC-ROC Score",
            value=f"{roc_auc:.3f}",
            help="Capacidade de distinguir entre classes (0.5 = aleatório, 1 = perfeito)"
        )
    
    st.markdown("---")
    
    # Simulador de risco
    st.subheader("Simulador de Risco de AVC")
    
    with st.form("prediction_form"):    
        col_a, col_b = st.columns(2)
        
        with col_a:
            idade = st.slider("Idade", 0, 130, 45)
            sexo = st.radio("Sexo", ['Homem', 'Mulher'])
            raca = st.selectbox("Raça/Cor", ['Branca', 'Preta', 'Amarela', 'Parda', 'Indígena'])
            plano_saude = st.radio("Possui plano de saúde?", ['Sim', 'Não'])
            peso = st.number_input("Peso (kg)", min_value=1.0, max_value=599.0, value=70.0)
            altura = st.number_input("Altura (cm)", min_value=1, max_value=299, value=170)
            consumo_sal = st.selectbox("Consumo de sal", ['Muito alto', 'Alto', 'Adequado', 'Baixo', 'Muito baixo'])
        
        with col_b:
            hipertensao = st.radio("Diagnóstico de hipertensão?", ['Sim', 'Não'])
            doenca_coracao = st.radio("Diagnóstico de doença cardíaca?", ['Sim', 'Não'])
            depressao = st.radio("Diagnóstico de depressão?", ['Sim', 'Não'])
            tabagismo = st.radio("Fuma atualmente?", ['Sim', 'Não'])
            consumo_alcool = st.selectbox("Frequência de consumo de álcool", ['Nunca', 'Menos de 1x/mês', '1x ou mais/mês'])
            exercicio = st.radio("Pratica exercício físico?", ['Sim', 'Não'])
            horas_tv = st.selectbox("Horas assistindo TV/dia", ['<1h', '1-2h', '2-3h', '3-6h', '6h+', 'Não assiste'])
        
        submitted = st.form_submit_button("🔍 Calcular Risco")
        
        if submitted:
            # Preparar dados para predição
            input_data = {
                'C008': idade,
                'C006_Mulher': 1 if sexo == 'Mulher' else 0,
                'C009_Preta': 1 if raca == 'Preta' else 0,
                'C009_Amarela': 1 if raca == 'Amarela' else 0,
                'C009_Parda': 1 if raca == 'Parda' else 0,
                'C009_Indígena': 1 if raca == 'Indígena' else 0,
                'I00102_Sim': 1 if plano_saude == 'Sim' else 0,
                'P00104': peso,
                'P00404': altura,
                'P02601_Muito alto': 1 if consumo_sal == 'Muito alto' else 0,
                'P02601_Alto': 1 if consumo_sal == 'Alto' else 0,
                'P02601_Adequado': 1 if consumo_sal == 'Adequado' else 0,
                'P02601_Baixo': 1 if consumo_sal == 'Baixo' else 0,
                'P027_1x ou mais/mês': 1 if consumo_alcool == '1x ou mais/mês' else 0,
                'P027_Menos de 1x/mês': 1 if consumo_alcool == 'Menos de 1x/mês' else 0,
                'Q00201_Sim': 1 if hipertensao == 'Sim' else 0,
                'Q06306_Sim': 1 if doenca_coracao == 'Sim' else 0,
                'Q092_Sim': 1 if depressao == 'Sim' else 0,
                'P050_Sim': 1 if tabagismo == 'Sim' else 0,
                'P034_Sim': 1 if exercicio == 'Sim' else 0,
                'P04501_2': 1 if horas_tv == '1-2h' else 0,
                'P04501_3': 1 if horas_tv == '2-3h' else 0,
                'P04501_4': 1 if horas_tv == '3-6h' else 0,
                'P04501_5': 1 if horas_tv == '6h+' else 0,
                'P04501_6': 1 if horas_tv == 'Não assiste' else 0
            }
            
            # Criar DataFrame com todas as features do modelo
            input_df = pd.DataFrame(columns=feature_names)
            input_df.loc[0] = 0  # Inicializar com zeros
            for key, value in input_data.items():
                if key in input_df.columns:
                    input_df[key] = value
            
            # Fazer predição
            try:
                proba = model.predict_proba(input_df)[0][1]
                risco = proba * 100
                
                # Mostrar resultado
                st.subheader("Resultado da Predição")
                
                if risco < 15:
                    st.success(f"Risco Baixo: {risco:.1f}% de chance de AVC")
                elif risco < 35:
                    st.warning(f"Risco Moderado: {risco:.1f}% de chance de AVC")
                else:
                    st.error(f"Risco Alto: {risco:.1f}% de chance de AVC")
                
                # Gráfico de risco
                fig_risk = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = risco,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Nível de Risco de AVC", 'font': {'color': 'white'}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickcolor': 'white'},
                        'bar': {'color': COLOR_PALETTE['accent']},
                        'steps': [
                            {'range': [0, 15], 'color': COLOR_PALETTE['success']},
                            {'range': [15, 35], 'color': COLOR_PALETTE['accent']},
                            {'range': [35, 100], 'color': COLOR_PALETTE['danger']}],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': risco}
                    }
                ))
                fig_risk.update_layout(paper_bgcolor=COLOR_PALETTE['card'], height=300)
                st.plotly_chart(fig_risk, use_container_width=True)
                
                st.subheader("Recomendações e Cuidados")
                
                st.write("""
                **Independente do seu nível de risco, estas medidas podem ajudar na prevenção:**
                """)
                
                col_rec1, col_rec2 = st.columns(2)
                
                with col_rec1:
                    st.write("""
                    🩺 **Monitoramento de saúde:**
                    - Verifique regularmente sua pressão arterial
                    - Faça exames de colesterol e glicemia anualmente
                    - Consulte um médico se tiver sintomas como tonturas ou dormência
                    """)
                    
                    st.write("""
                    🥗 **Hábitos alimentares:**
                    - Reduza o consumo de sal e alimentos processados
                    - Aumente a ingestão de frutas, verduras e grãos integrais
                    - Mantenha-se hidratado
                    """)
                
                with col_rec2:
                    st.write("""
                    🏃 **Atividade física:**
                    - Pratique pelo menos 150 minutos de exercícios moderados por semana
                    - Evite ficar sentado por longos períodos
                    - Incorpore atividades simples como caminhadas no dia a dia
                    """)
                    
                    st.write("""
                    🚭 **Hábitos saudáveis:**
                    - Não fume e evite ambientes com fumo
                    - Limite o consumo de álcool
                    - Mantenha um peso saudável
                    - Durma bem (7-8 horas por noite)
                    """)
                
                # Recomendações específicas baseadas no perfil
                st.subheader("Recomendações Personalizadas")
                rec_especificas = []
                
                if hipertensao == 'Sim':
                    rec_especificas.append("🔹 **Hipertensão:** Controle rigoroso da pressão arterial é essencial. Siga as orientações médicas e tome os medicamentos conforme prescrito.")
                
                if tabagismo == 'Sim':
                    rec_especificas.append("🔹 **Tabagismo:** Parar de fumar reduz significativamente o risco de AVC. Considere programas de cessação do tabagismo.")
                
                if consumo_sal in ['Muito alto', 'Alto']:
                    rec_especificas.append("🔹 **Consumo de sal:** Reduza a ingestão de sal para menos de 5g por dia (cerca de 1 colher de chá).")
                
                if exercicio == 'Não':
                    rec_especificas.append("🔹 **Exercício físico:** Comece com atividades leves como caminhadas de 30 minutos, 5 vezes por semana.")
                
                if idade > 50:
                    rec_especificas.append("🔹 **Idade:** Como você tem mais de 50 anos, check-ups anuais são especialmente importantes.")
                
                if doenca_coracao == 'Sim':
                    rec_especificas.append("🔹 **Doença cardíaca:** Siga rigorosamente o tratamento prescrito pelo seu cardiologista.")
                
                if len(rec_especificas) > 0:
                    for rec in rec_especificas:
                        st.write(rec)
                else:
                    st.info("Seu perfil não apresentou fatores de risco significativos que necessitem de recomendações específicas além das gerais.")
                
                st.write("""
                **Lembre-se:** Esta avaliação não substitui uma consulta médica. Consulte regularmente seu médico para avaliações personalizadas.
                """)
                
            except Exception as e:
                st.error(f"Erro ao fazer a predição: {str(e)}")
                st.write("Por favor, verifique os dados inseridos e tente novamente.")

# Página 3: Sobre o Projeto
# Página 3: Sobre o Projeto - Versão Corrigida
else:
    st.title("ℹ️ Sobre o Projeto")
    
    with st.container():
        st.markdown(f"""
        <style>
            .info-card {{
                background-color: {COLOR_PALETTE['card']};
                padding: 1px;
                border-radius: 10px;
                margin-bottom: 20px;
            }}
        </style>
        <div class="info-card">
        """, unsafe_allow_html=True)
        
        st.header("Data Science Aplicado à Saúde")
        st.subheader("Projeto: Análise de Fatores de Risco para AVC")
        
        st.write("""
        Este trabalho foi desenvolvido como parte do projeto de extensão **Data Science Aplicado à Saúde**, 
        com foco específico na análise preditiva de Acidente Vascular Cerebral (AVC).
        """)
        
        st.write("""
        **Objetivo do meu projeto:**
        Desenvolver uma ferramenta de análise e predição de risco de AVC baseada em dados epidemiológicos da 
        Pesquisa Nacional de Saúde, identificando os principais fatores de risco na população brasileira.
        """)
        
        st.subheader("Metodologia Científica")
        st.markdown("""
        - **Base de dados:** Pesquisa Nacional de Saúde (PNS) 2019
        - **Técnicas:** Análise exploratória, modelagem preditiva (Random Forest)
        - **Visualização:** Dashboard interativo com Streamlit
        - **Público-alvo:** Profissionais de saúde e gestores públicos
        """)
        
        st.subheader("Créditos")
        st.markdown("""
        - **Orientador:** Prof. Dr. Marcelo Henklain
        - **Desenvolvedor:** Kaylon Gutierre Peres Gonçalves
        - **Instituição:** Universidade Federal de Roraima
        """)
        
        st.markdown("</div>", unsafe_allow_html=True)