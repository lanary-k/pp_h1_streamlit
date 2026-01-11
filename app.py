import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import time
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures


st.set_page_config(layout="wide")

seasons = {
    12: 'winter', 1: 'winter', 2: 'winter', 
    3: 'spring', 4: 'spring', 5: 'spring', 
    6: 'summer', 7: 'summer', 8: 'summer', 
    9: 'autumn', 10: 'autumn', 11: 'autumn'
}

if 'events_history' not in st.session_state:
    st.session_state.events_history = pd.DataFrame(
        columns=['timestamp', 'event_name', 'event_category', 'duration']
    )

events_history = st.session_state.events_history

def log_event(event_name, event_category, duration):
    new_event = {
        'timestamp': datetime.now(),
        'event_name': event_name,
        'event_category': event_category,
        'duration': duration
    }
    
    st.session_state.events_history = pd.concat(
        [st.session_state.events_history, pd.DataFrame([new_event])], 
        ignore_index=True
    )

def curr_season_temp_plot(current_temp, mean_temp, lower, upper, season_name):
    """
    Визуализация текущей температуры отностительно нормы сезона
    """

    seasons_color = {
      'winter' : 'lightblue',
      'spring' : 'lightgreen',
      'summer' : 'lightcoral',
      'autumn' : 'sandybrown'
    }

    min_show = min(lower, current_temp) - 3
    max_show = max(upper, current_temp) + 3
    
    fig = go.Figure()

    # Фон
    fig.add_shape(
        x0=lower,
        x1=upper,
        fillcolor=seasons_color[season_name],
        line_width=0,
        opacity=0.3
    )
    
    #Средняя температура
    fig.add_shape(
        type="line",
        x0=mean_temp,
        x1=mean_temp,
        line=dict(color="black", width=2),
        name=f"Средняя: {mean_temp:.1f}°C"
    )

    fig.add_annotation(
        x=mean_temp,
        y=0.2,
        text=f"Средняя {mean_temp:.1f}°C",
        showarrow=False,
        font=dict(size=10),
        yshift=-10
    )
    
    # Текущая температура
    fig.add_trace(go.Scatter(
        x=[current_temp],
        y=[0.5],
        mode="markers",
        marker=dict(
            color="black",
            size=15
        ),
        hovertemplate="<b>Текущая температура</b><br>" +
                     f"Значение: {current_temp:.1f}°C<br>" +
                     "<extra></extra>"
        
    ))

    fig.add_annotation(
        x=current_temp,
        y=0.65,
        text=f"<b>Сейчас<br>{current_temp:.1f}°C</b>",
        showarrow=False,
        font=dict(size=12, color="black"),
        yshift=10
    )
    
    # Границы диапазона
    fig.add_annotation(
        x=lower,
        y=0.75,
        text=f"{lower:.1f}°C",
        showarrow=False,
        yshift=10
    )
    
    fig.add_annotation(
        x=upper,
        y=0.75,
        text=f"{upper:.1f}°C",
        showarrow=False,
        yshift=10
    )
    
    # Настройка макета
    fig.update_layout(
        title=dict(
            #text=f"Температура сегодня относительно исторической нормы за сезон {season_name.capitalize()}",
            text=f"Температура сегодня относительно исторической нормы за сезон",
            font=dict(size=18, weight="bold"),
        ),
        xaxis=dict(
            gridcolor="rgba(0,0,0,0.1)",
            zeroline=False
        ),
        yaxis=dict(
            range=[0, 1],
            showticklabels=False,
        ),
        height=300,
        width=800,
        showlegend=False,
        plot_bgcolor="white"
    )
    return fig

def get_current_temperature(city_name, api_key):
  url = "http://api.openweathermap.org/data/2.5/weather"
  params = {
        'q': city_name,
        'appid': api_key,
        'units': 'metric'
  }
  try:
    response = requests.get(url, params=params)
    weather_data = response.json()
    return weather_data
  except Exception as e:
        return {"error": str(e), "cod": 500}

async def get_current_temperature_async(city_name, api_key):
  url = "http://api.openweathermap.org/data/2.5/weather"
  params = {
        'q': city_name,
        'appid': api_key,
        'units': 'metric'
  }
  try:
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params, timeout=10) as response:
            return await response.json()
  except Exception as e:
        return {"error": str(e), "cod": 500}

    


with st.sidebar:
    st.title("Исторические данные")
    history_file = st.file_uploader('Выберите файл', type='csv')

    use_parallel = st.checkbox("Параллельное вычсление метрик")
    use_acync = st.checkbox("Асинхронное получание данных с OpenWeatherMap")

st.title("Анализ температурных данных и мониторинг текущей температуры")

if history_file is not None:
    df_ex = pd.read_csv(history_file)

    with st.sidebar:
        st.title("Настройки")

        choise_city = st.selectbox(
            "Выберите город:",
            df_ex['city'].unique(),
            index=0
        )

        period = st.selectbox(
            "Исторические данные за:",
            ['Весь период', 'Последние 5 лет', 'Последние 2 года', 'Последний год', 'Последний месяц'],
            index=2
        )
    
    df = df_ex[df_ex['city'] == choise_city]

    current_date = datetime.now()

    if period == 'Последний год':
        first_zoom_date = current_date - pd.Timedelta(days=365)
    elif period == 'Последние 5 лет':
        first_zoom_date = current_date - pd.Timedelta(days=5*365)
    elif period == 'Последние 2 года':
        first_zoom_date = current_date - pd.Timedelta(days=2*365)
    elif period == 'Последний месяц':
        first_zoom_date = current_date - pd.Timedelta(days=30)
    else:
        first_zoom_date = df['timestamp'].min()
    
    # Расчет метрик

    # Распараллеливание по городам
    if use_parallel:
        t1 = time.time()
        def process_city(df_city):
            df_city = df_city.copy()
            df_city = df_city.sort_values(['city', 'timestamp'])
            
            df_city['temp_ma'] = df_city['temperature'].rolling(window=30, min_periods=1).mean()
            df_city['mean'] = df_city.groupby('season')['temperature'].transform('mean')
            df_city['std'] = df_city.groupby('season')['temperature'].transform('std')
            df_city['l1'] = df_city['mean'] - 2 * df_city['std']
            df_city['l2'] = df_city['mean'] + 2 * df_city['std']
            df_city['is_anomaly'] = (df_city['temperature'] < df_city['l1']) | (df_city['temperature'] > df_city['l2'])
            return df_city
        
        city_groups = [group for _, group in df.groupby('city')]
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            processed = list(executor.map(process_city, city_groups))
        
        df = pd.concat(processed, ignore_index=True)
        t2 = time.time()
        print(f"Время вычислений с распараллеливанием: {t2-t1}")
        log_event('Вычисление метрик', 'С распараллеливанием', t2-t1)


    # Расчет метрик сразу для всех городов
    else:
        t1 = time.time()
        df = df.sort_values(['city', 'timestamp'])
        df['temp_ma'] = df.groupby('city')['temperature'].transform(
            lambda x: x.rolling(window=30, min_periods=1).mean()
        )
        df['mean'] = df.groupby(['city', 'season'])['temperature'].transform(lambda x: x.mean())
        df['std'] = df.groupby(['city', 'season'])['temperature'].transform(lambda x: x.std())
        df['l1'] = df['mean'] - 2 * df['std']
        df['l2'] = df['mean'] + 2 * df['std']
        df['is_anomaly'] = (df['temperature'] < df['l1']) | (df['temperature'] > df['l2'])
        t2 = time.time()
        print(f"Время вычислений без распараллеливания: {t2-t1}")
        log_event('Вычисление метрик', 'Без распараллеливания', t2-t1)


    # Построение графика за исторический период
    fig = go.Figure()
    fig = px.scatter(
        df, x='timestamp', y='temperature',
        color='is_anomaly',
        color_discrete_sequence=['blue', 'red'],
        labels={'is_anomaly': 'Наблюдения'},
        
    )
    fig.data[0].name = "Нормальные"
    fig.data[1].name = "Аномалии"

    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['l2'],
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['l1'],
        line=dict(width=0),
        fill='tonexty',
        fillcolor='#b7bbf7',
        name='ДИ для сезона'
    ))


    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['temp_ma'],
        mode='lines',
        line=dict(color='black', width=3),
        name='Тренд'
    ))

    fig.update_layout(
        title=f'Исторические данные для города {choise_city}',
        xaxis_title='Дни',
        yaxis_title='Температура'
    )

    fig.update_layout(
        xaxis=dict(
            range=[first_zoom_date, current_date]  
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Статистика по сезонам")
    season_stats = df.groupby('season')['temperature'].agg([
        'mean', 'std', 'min', 'max', 'count'
    ]).round(2)
    st.dataframe(season_stats)

    with st.expander("Показать детали"):
        st.dataframe(df.head(100))

    st.subheader("Текущая температура в городе")
    api_key = st.text_input("Для отображения текущей погоды введите API-ключ OpenWeatherMap")

    if api_key != '':
        try:
            if use_acync:
                t1 = time.time()
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                start_time = time.time()
                weather_info = loop.run_until_complete(get_current_temperature_async(choise_city, api_key))
                t2 = time.time()
                print(f"Время синхронного запуска: {t2-t1}")
                log_event('Запросы к OpenWeatherMap', 'Асинхронно', t2-t1)

            else:
                t1 = time.time()
                weather_info = get_current_temperature(choise_city, api_key)
                t2 = time.time()
                print(f"Время асинхронного запуска: {t2-t1}")
                log_event('Запросы к OpenWeatherMap', 'Синхронно', t2-t1)
            
            curr_temp = weather_info.get('main').get('temp')
            curr_season = seasons[current_date.month]

            mean_by_season = df[df['season'] == curr_season]['mean'].mean()
            l1_by_season = df[df['season'] == curr_season]['l1'].mean()
            l2_by_season = df[df['season'] == curr_season]['l2'].mean()

            fig = curr_season_temp_plot(
                current_temp=curr_temp,
                mean_temp=mean_by_season,
                lower=l1_by_season,
                upper=l2_by_season,
                season_name=curr_season
            )

            col1, col2 = st.columns([0.25, 0.75])
            with col1:
                st.title("")
                st.metric("Температура сегодня", f"{curr_temp:.1f}°C")
                if l1_by_season <= curr_temp and curr_temp <= l2_by_season:
                    st.text("В пределах нормы для сезона")
                else:
                    st.text("За пределами для сезона")
            with col2:
                st.plotly_chart(fig, use_container_width=True)
        except:
            if 'cod' in weather_info and weather_info['cod'] == 401:
                st.error("Неверный API-ключ")
            else:
                st.error(weather_info)
    else:
        pass

with st.expander("Показать историю запросов (технический лог)"):
    col1, col2 = st.columns(2)
    with col1:
        compute_data = events_history[events_history['event_name'] == 'Вычисление метрик']
        if not compute_data.empty:
            compute_avg = compute_data.groupby('event_category')['duration'].mean().reset_index()
            
            fig1 = px.bar(compute_avg, x='event_category', y='duration',
                          title='Вычисление метрик (среднее время)',
                          labels={'duration': 'Среднее время (сек)', 'event_category': 'Метод'},
                          text_auto='.3f',
                          color='event_category')
            
            fig1.update_traces(
                hovertemplate='<b>%{x}</b><br>Среднее время: %{y:.3f} сек<extra></extra>'
            )
            st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        api_data = events_history[events_history['event_name'] == 'Запросы к OpenWeatherMap']
        if not api_data.empty:
            api_avg = api_data.groupby('event_category')['duration'].mean().reset_index()
            
            fig2 = px.bar(api_avg, x='event_category', y='duration',
                          title='Запросы к OpenWeatherMap (среднее время)',
                          labels={'duration': 'Среднее время (сек)', 'event_category': 'Метод'},
                          text_auto='.3f',
                          color='event_category')
            
            fig2.update_traces(
                hovertemplate='<b>%{x}</b><br>Среднее время: %{y:.3f} сек<extra></extra>'
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(events_history)