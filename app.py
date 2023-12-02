import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components
import folium
from folium.plugins import HeatMap

# 데이터 로딩
@st.cache_data
def load_data():
    return pd.read_csv('sample_analysis.csv')
df_analysis = load_data()
df_analysis = df_analysis.drop(columns='Unnamed: 0')

# 사이드바 설정
st.sidebar.title('메뉴')
page = st.sidebar.radio('페이지 선택', ['메인 페이지', '비교 분석', '상관 분석', 'GPS좌표'])


# 기본 페이지
if page == '메인 페이지':
    # Streamlit 제목 설정
    st.title('자전거 도로 주행 분석')
    st.write('')

    # 필터링 옵션
    st.write('필터링 옵션 (디폴트는 전체선택)')
    selected_place = st.multiselect('장소 선택', options=df_analysis['place'].unique())
    selected_daynight = st.multiselect('주간/야간 선택', options=df_analysis['daynight'].unique())
    selected_weather = st.multiselect('날씨 선택', options=df_analysis['weather'].unique())
    selected_bike_lane = st.multiselect('자전거 도로 유형 선택', options=df_analysis['bike_lane'].unique())
    selected_date_range = st.date_input('날짜 범위 선택', [])

    # 필터링된 데이터
    filtered_df = df_analysis.copy()
    if selected_place:
        filtered_df = filtered_df[filtered_df['place'].isin(selected_place)]
    if selected_daynight:
        filtered_df = filtered_df[filtered_df['daynight'].isin(selected_daynight)]
    if selected_weather:
        filtered_df = filtered_df[filtered_df['weather'].isin(selected_weather)]   
    if selected_bike_lane:
        filtered_df = filtered_df[filtered_df['bike_lane'].isin(selected_bike_lane)]
    if selected_date_range:
        filtered_df = filtered_df[(filtered_df['date_captured'] >= selected_date_range[0]) & 
                                (filtered_df['date_captured'] <= selected_date_range[1])]

    st.write('')
    # 기준 별 자전거 이용량
    st.header('기준별 자전거 이용량')
    # 필터링된 옵션을 카피 (이래야 필터링 바꿔도 초기화)
    df_analysis = filtered_df.copy()
    # date_captured 열을 datetime 타입으로 변환
    df_analysis['date_captured'] = pd.to_datetime(df_analysis['date_captured'])
    df_analysis['hour'] = df_analysis['date_captured'].dt.hour
    df_analysis['weekday'] = df_analysis['date_captured'].dt.weekday
    df_analysis['month'] = df_analysis['date_captured'].dt.month

    # 사용자 입력 받기
    option = st.selectbox('분석할 기준을 선택하세요', ['시간별', '요일별', '월별', 'day/night별', '날씨별', '장소별', '자전거 도로 유형별'])

    # 분석 및 시각화
    if option == '시간별':
        usage = df_analysis.groupby('hour').size()
        fig, ax = plt.subplots()
        sns.barplot(x=usage.index, y=usage.values, ax=ax)
        ax.set_xlabel('Hour of the Day')
        ax.set_ylabel('Number of Uses')
        st.pyplot(fig)

    elif option == '요일별':
        usage = df_analysis.groupby('weekday').size()
        fig, ax = plt.subplots()
        sns.barplot(x=usage.index, y=usage.values, ax=ax)
        ax.set_xlabel('Day of the Week')
        ax.set_ylabel('Number of Uses')
        ax.set_xticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        st.pyplot(fig)

    elif option == '월별':
        usage = df_analysis.groupby('month').size()
        fig, ax = plt.subplots()
        sns.barplot(x=usage.index, y=usage.values, ax=ax)
        ax.set_xlabel('Month')
        ax.set_ylabel('Number of Uses')
        st.pyplot(fig)

    elif option == 'day/night별':
        usage = df_analysis.groupby('daynight').size()
        fig, ax = plt.subplots()
        sns.barplot(x=usage.index, y=usage.values, ax=ax)
        ax.set_xlabel('day or night')
        ax.set_ylabel('Number of Uses')

    elif option == '날씨별':
        usage = df_analysis.groupby('weather').size()
        fig, ax = plt.subplots()
        sns.barplot(x=usage.index, y=usage.values, ax=ax)
        ax.set_xlabel('weather')
        ax.set_ylabel('Number of Uses')

    elif option == '장소별':
        usage = df_analysis.groupby('place').size()
        fig, ax = plt.subplots()
        sns.barplot(x=usage.index, y=usage.values, ax=ax)
        ax.set_xlabel('place')
        ax.set_ylabel('Number of Uses')
        st.pyplot(fig)

    elif option == '자전거 도로 유형별':
        usage = df_analysis.groupby('bike_lane').size()
        fig, ax = plt.subplots()
        sns.barplot(x=usage.index, y=usage.values, ax=ax)
        ax.set_xlabel('bike_lane')
        ax.set_ylabel('Number of Uses')
        st.pyplot(fig)




# 비교 분석 페이지
elif page == '비교 분석':
    st.title('다른 옵션으로 필터링한 데이터 비교')
    st.write('')

    # 두 개의 열 생성
    col1, col2 = st.columns(2)

    # 필터링 옵션 세트 1
    with col1:
        st.write('필터링 옵션 세트 1 (디폴트는 전체선택)')
        selected_place_1 = st.multiselect('장소 선택', options=df_analysis['place'].unique(), key='place1')
        selected_daynight_1 = st.multiselect('주간/야간 선택', options=df_analysis['daynight'].unique(), key='daynight1')
        selected_weather_1 = st.multiselect('날씨 선택', options=df_analysis['weather'].unique(), key='weather1')
        selected_bike_lane_1 = st.multiselect('자전거 도로 유형 선택', options=df_analysis['bike_lane'].unique(), key='bike_lane1')
        selected_date_range_1 = st.date_input('날짜 범위 선택', [], key='date_range1')

        # 필터링된 데이터
        filtered_df_1 = df_analysis.copy()
        if selected_place_1:
            filtered_df_1 = filtered_df_1[filtered_df_1['place'].isin(selected_place_1)]
        if selected_daynight_1:
            filtered_df_1 = filtered_df_1[filtered_df_1['daynight'].isin(selected_daynight_1)]
        if selected_weather_1:
            filtered_df_1 = filtered_df_1[filtered_df_1['weather'].isin(selected_weather_1)]   
        if selected_bike_lane_1:
            filtered_df_1 = filtered_df_1[filtered_df_1['bike_lane'].isin(selected_bike_lane_1)]
        if selected_date_range_1:
            filtered_df_1 = filtered_df_1[(filtered_df_1['date_captured'] >= selected_date_range_1[0]) & 
                                    (filtered_df_1['date_captured'] <= selected_date_range_1[1])]
            
    # 필터링 옵션 세트 2
    with col2:
        st.write('필터링 옵션 세트 2 (디폴트는 전체선택)')
        selected_place_2 = st.multiselect('장소 선택', options=df_analysis['place'].unique(), key='place2')
        selected_daynight_2 = st.multiselect('주간/야간 선택', options=df_analysis['daynight'].unique(), key='daynight2')
        selected_weather_2 = st.multiselect('날씨 선택', options=df_analysis['weather'].unique(), key='weather2')
        selected_bike_lane_2 = st.multiselect('자전거 도로 유형 선택', options=df_analysis['bike_lane'].unique(), key='bike_lane2')
        selected_date_range_2 = st.date_input('날짜 범위 선택', [], key='date_range2')

        # 필터링된 데이터
        filtered_df_2 = df_analysis.copy()
        if selected_place_2:
            filtered_df_2 = filtered_df_2[filtered_df_2['place'].isin(selected_place_2)]
        if selected_daynight_2:
            filtered_d_2 = filtered_df_2[filtered_df_2['daynight'].isin(selected_daynight_2)]
        if selected_weather_2:
            filtered_df_2 = filtered_df_2[filtered_df_2['weather'].isin(selected_weather_2)]   
        if selected_bike_lane_2:
            filtered_df_2 = filtered_df_2[filtered_df_2['bike_lane'].isin(selected_bike_lane_2)]
        if selected_date_range_2:
            filtered_df_2 = filtered_df_2[(filtered_df_2['date_captured'] >= selected_date_range_2[0]) & 
                                    (filtered_df_2['date_captured'] <= selected_date_range_2[1])]
    
    st.write('')
    # 기준 별로 데이터 분석
    st.header('두 집단 기준별로 비교')
    # 필터링된 옵션을 카피 (이래야 필터링 바꿔도 초기화)
    df_analysis_1 = filtered_df_1.copy()
    df_analysis_2 = filtered_df_2.copy()
    # date_captured 열을 datetime 타입으로 변환
    for df_analysis in [df_analysis_1, df_analysis_2]:
        df_analysis['date_captured'] = pd.to_datetime(df_analysis['date_captured'])
        df_analysis['hour'] = df_analysis['date_captured'].dt.hour
        df_analysis['weekday'] = df_analysis['date_captured'].dt.weekday
        df_analysis['month'] = df_analysis['date_captured'].dt.month
    # 사용자 입력 받기
    option = st.selectbox('분석할 기준을 선택하세요', ['시간별', '요일별', '월별', 'day/night별', '날씨별', '장소별', '자전거 도로 유형별'])

    # 분석 및 시각화
    if option == '시간별':
        fig, axs = plt.subplots(2,1)
        # 필터링1
        usage_1 = df_analysis_1.groupby('hour').size()
        sns.barplot(x=usage_1.index, y=usage_1.values, ax=axs[0])
        axs[0].set_xlabel('Hour of the Day')
        axs[0].set_ylabel('Number of Uses')
        # 필터링2
        usage_2 = df_analysis_2.groupby('hour').size()
        sns.barplot(x=usage_2.index, y=usage_2.values, ax=axs[1])
        axs[1].set_xlabel('Hour of the Day')
        axs[1].set_ylabel('Number of Uses')
        # 서브플롯 간격 조정
        plt.subplots_adjust(hspace=0.5)
        st.pyplot(fig)

    elif option == '요일별':
        fig, axs = plt.subplots(2,1)
        # 필터링1
        usage_1 = df_analysis_1.groupby('weekday').size()
        sns.barplot(x=usage_1.index, y=usage_1.values, ax=axs[1])
        axs[0].set_xlabel('Day of the Week')
        axs[0].set_ylabel('Number of Uses')
        axs[0].set_xticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        # 필터링2
        usage_2 = df_analysis_2.groupby('weekday').size()
        sns.barplot(x=usage_2.index, y=usage_2.values, ax=axs[0])
        axs[1].set_xlabel('Day of the Week')
        axs[1].set_ylabel('Number of Uses')
        axs[1].set_xticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        # 서브플롯 간격 조정
        plt.subplots_adjust(hspace=0.5)
        st.pyplot(fig)

    elif option == '월별':
        fig, axs = plt.subplots(2,1)
        # 필터링1
        usage_1 = df_analysis_1.groupby('month').size()
        sns.barplot(x=usage_1.index, y=usage_1.values, ax=axs[0])
        axs[0].set_xlabel('Month')
        axs[0].set_ylabel('Number of Uses')
        # 필터링2
        usage_2 = df_analysis_2.groupby('month').size()
        sns.barplot(x=usage_2.index, y=usage_2.values, ax=axs[1])
        axs[1].set_xlabel('Month')
        axs[1].set_ylabel('Number of Uses')
        # 서브플롯 간격 조정
        plt.subplots_adjust(hspace=0.5)
        st.pyplot(fig)

    elif option == 'day/night별':
        fig, axs = plt.subplots(2,1)
        # 필터링1
        usage_1 = df_analysis_1.groupby('daynight').size()
        sns.barplot(x=usage_1.index, y=usage_1.values, ax=axs[0])
        axs[0].set_xlabel('day or night')
        axs[0].set_ylabel('Number of Uses')
        # 필터링2
        usage_2 = df_analysis_2.groupby('daynight').size()
        sns.barplot(x=usage_2.index, y=usage_2.values, ax=axs[1])
        axs[1].set_xlabel('day or night')
        axs[1].set_ylabel('Number of Uses')
        # 서브플롯 간격 조정
        plt.subplots_adjust(hspace=0.5)
        st.pyplot(fig)

    elif option == '날씨별':
        fig, axs = plt.subplots(2,1)
        # 필터링1
        usage_1 = df_analysis_1.groupby('weather').size()
        sns.barplot(x=usage_1.index, y=usage_1.values, ax=axs[0])
        axs[0].set_xlabel('weather')
        axs[0].set_ylabel('Number of Uses')
        # 필터링2
        usage_2 = df_analysis_2.groupby('weather').size()
        sns.barplot(x=usage_2.index, y=usage_2.values, ax=axs[1])
        axs[1].set_xlabel('weather')
        axs[1].set_ylabel('Number of Uses')
        # 서브플롯 간격 조정
        plt.subplots_adjust(hspace=0.5)
        st.pyplot(fig)

    elif option == '장소별':
        fig, axs = plt.subplots(2,1)
        # 필터링1
        usage_1 = df_analysis_1.groupby('place').size()
        sns.barplot(x=usage_1.index, y=usage_1.values, ax=axs[0])
        axs[0].set_xlabel('place')
        axs[0].set_ylabel('Number of Uses')
        # 필터링2
        usage_2 = df_analysis_2.groupby('place').size()
        sns.barplot(x=usage_2.index, y=usage_2.values, ax=axs[1])
        axs[1].set_xlabel('place')
        axs[1].set_ylabel('Number of Uses')
        # 서브플롯 간격 조정
        plt.subplots_adjust(hspace=0.5)
        st.pyplot(fig)

    elif option == '자전거 도로 유형별':
        fig, axs = plt.subplots(2,1)
        # 필터링1
        usage_1 = df_analysis_1.groupby('bike_lane').size()
        sns.barplot(x=usage_1.index, y=usage_1.values, ax=axs[0])
        axs[0].set_xlabel('bike_lane')
        axs[0].set_ylabel('Number of Uses')
        # 필터링2
        usage_2 = df_analysis_2.groupby('bike_lane').size()
        sns.barplot(x=usage_2.index, y=usage_2.values, ax=axs[1])
        axs[1].set_xlabel('bike_lane')
        axs[1].set_ylabel('Number of Uses')
        # 서브플롯 간격 조정
        plt.subplots_adjust(hspace=0.5)
        st.pyplot(fig)


# 상관 분석 페이지
elif page == '상관 분석':
    st.title("자전거 도로 주행 데이터 상관 분석")
    st.write('')

    # 수치형 데이터를 포함하는 열만 선택 가능하도록 multiselect 위젯 추가
    numeric_columns = df_analysis.select_dtypes(include=[np.number]).columns

    # '전체 선택'을 포함하는 선택 옵션 만들기
    options = ['전체 선택'] + list(numeric_columns)

    # multiselect 위젯 추가, 기본적으로 아무것도 선택되지 않도록 설정
    selected_options = st.multiselect('분석할 열을 선택하세요:', options)

    # '전체 선택'이 선택된 경우 모든 열을 선택
    if '전체 선택' in selected_options:
        selected_columns = list(numeric_columns)
    else:
        selected_columns = selected_options

    # 선택된 열이 있을 경우 상관관계 계산
    if selected_columns:
        # 선택된 열에 대한 상관관계 계산
        correlation = df_analysis[selected_columns].corr()
        # 상관관계 표시
        st.write('상관관계:', correlation)
        # 상관관계를 히트맵으로 시각화
        fig, ax = plt.subplots()  # 새로운 Figure와 Axes 생성
        sns.heatmap(correlation, annot=True, ax=ax, square=True)  # 히트맵을 Axes 객체에 그리기
        # x축 레이블을 상단으로 이동
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top') 
        st.pyplot(fig)  # Figure 객체를 Streamlit에 전달
    
        # 사용자로부터 상관계수 임계값 입력받기
        correlation_threshold = st.number_input('상관계수 임계값을 입력하세요 (예: 0.3)', min_value=0.0, max_value=1.0, value=0.3, step=0.1)

        # 상관관계가 입력된 임계값 이상인 관계들만 추출하여 표시
        high_corr = correlation.abs() >= correlation_threshold
        high_corr_pairs = correlation.where(high_corr).stack().reset_index()
        high_corr_pairs.columns = ['Variable 1', 'Variable 2', 'Correlation']
        high_corr_pairs = high_corr_pairs[high_corr_pairs['Variable 1'] != high_corr_pairs['Variable 2']]

        st.write(f'상관계수의 절대값이 {correlation_threshold:.2f} 이상인 관계:', high_corr_pairs)


# GPS좌표
elif page == 'GPS좌표':

    # 필터링 옵션
    st.write('지역 선택 (디폴트는 전체선택)')
    selected_place = st.multiselect('지역 선택', options=df_analysis['place'].unique())

    # 필터링된 데이터
    filtered_gps = df_analysis.copy()
    if selected_place:
        filtered_gps = filtered_gps[filtered_gps['place'].isin(selected_place)]

    # 중앙 위치 계산
    center_lat = filtered_gps['Latitude'].mean()
    center_lon = filtered_gps['Longitude'].mean()

    # 지도 생성
    map = folium.Map(location=[center_lat, center_lon], zoom_start=10)

    # 히트맵 추가
    HeatMap(data=filtered_gps[['Latitude', 'Longitude']], radius=10).add_to(map)

    # 지도를 HTML 문자열로 변환
    map_html = map._repr_html_()

    # Streamlit 앱에 지도 표시
    components.html(map_html, width=700, height=500)

