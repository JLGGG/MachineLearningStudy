# Artificial Intelligence Overview
***
Artificial Intelligence : 인간의 지능(인지, 추론, 학습등)을 컴퓨터나 시스템 등으로 만든 것 또는 만들 수 있는 방법론이나 실현 가능성 등을 연구하는 기술 또는 과학

Weak AI : 특정 문제의 해결   
Strong AI : 사람처럼 사고   

인공지능의 연구분야 
* 전문가시스템
* 자연어처리기
* 로보틱스
* 인공비전
* 인공신경망
* 지능에이전트
* 퍼지로직
* 유전자알고리즘

AI가 ML(Machine Learning)을 포함하고, ML이 DL(Deep Learning)을 포함한다. 현재 사회에서는 DL이 인공지능을 실현하는 방법론으로 여겨지고 있다.   

### 일반 프로그래밍 vs. 머신러닝
General Programming은 input으로 data, program을 입력하고 그것에 대한 output을 출력하지만, ML에서는 input으로 data, output을 입력하고 그것에 대한output으로 program(model)을 출력한다.   

### 머신러닝의 3대 원리
* Occam's Razor   
  같은 현상을 설명하는 두가지 모형이 있다면, 단순한 모형을 선택("All things being equal, the simplest solution tends to be the best one.")
* Sampling Bias   
  모집단을 대표성의 원리에 따라 표본을 추출하지 못할 때 기계학습 알고리즘도 편향된 표본을 학습하여 결과를 왜곡시킴
* Data Snooping Bias   
  데이터를 본 후 기계학습 알고리즘을 결정하는 것으로 기계학습 알고리즘은 데이터를 보기 전에 선정해야 함.   

### 머신러닝 분류   
* Supervised Learning : Regression, Classification
* Unsupervised Learning : Clustering(Ex. K-means...)
* Reinforcement Learning : Algorithm learns to react to an environment.

### 머신러닝 모델 개발 단계   
1. 데이터 수집과정
2. 데이터 가공과정
3. 데이터 학습 방법 선택
4. 매개변수 조정
5. 머신러닝 모델 학습
6. 머신러닝 모델 개발   
3~5 과정이 데이터 학습 과정이다.

### 머신러닝 모델 활용 단계   
실제 데이터와 개발된 머신러닝 모델을 실행해서 분류/결과값 도출 or 예측결과를 도출한다.

#### The curse of dimensionality   
입력변수의 차원이 증가할수록, 공간의 부피가 기하급수적으로 증가하고 데이터는 공간에 희소해져 데이터의 분포 분석이나 모델 추정에 필요한 샘플 데이터 개수가 기하급수적으로 증가.
![cod](https://user-images.githubusercontent.com/18206655/89172882-5385c680-d5be-11ea-87df-eb9b3419a1e6.jpg)


