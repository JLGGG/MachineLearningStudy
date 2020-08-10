# Artificial Intelligence Overview
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
<img src="https://user-images.githubusercontent.com/18206655/89280606-d2910280-d683-11ea-8d40-d39eccb1dc98.jpg" width=90%></img>

### 일반 프로그래밍 vs. 머신러닝
General Programming은 input으로 data, program을 입력하고 그것에 대한 output을 출력하지만, ML에서는 input으로 data, output을 입력하고 그것에 대한output으로 program(model)을 출력한다.   
기존의 코딩 방법이 조건을 라인바이라인으로 긴 코드로 써내려 갔다면(if-else 지옥!) 머신러닝의 코딩 방법은 조건을 학습 모델의 여러 가중치로 변환하는 일이다.

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
이론적인 과정:   
1. 데이터 수집과정
2. 데이터 가공과정
3. 데이터 학습 방법 선택
4. 매개변수 조정
5. 머신러닝 모델 학습
6. 머신러닝 모델 개발   
3~5 과정이 데이터 학습 과정이다. 머신러닝 모델을 개발하고 학습하는 시간보다 데이터를 전처리하는데 훨씬 더 많은 시간이 소요된다.   

실제 개발할 때 과정:   
1. 데이터 전처리: Pandas를 사용해서 데이터를 가공하고(데이터에 문자열이 있으면 안된다. 오로지 숫자만!), 가공된 데이터를 Numpy를 사용해서 Numpy 배열로 만든다.(train data 80%, test data 20%), 단위 dataFrame(df)
2. 모델설계: DNN, CNN(Image), RNN(Sequence), 몇층?, 뉴런의 수?
3. 모델훈련: model.fit(훈련 데이터 x, 훈련 데이터 y, ...) -> 훈련 정확도
4. 모델평가: model.evaluate(테스트 데이터 x, 테스트 데이터 y) -> 테스트 정확도
5. 모델예측: model.predict(새로운 데이터 x)   
* 알고리즘보다 중요한게 데이터 전처리이다

### 머신러닝 모델 활용 단계   
실제 데이터와 개발된 머신러닝 모델을 실행해서 분류/결과값 도출 or 예측결과를 도출한다.

#### The curse of dimensionality   
입력변수의 차원이 증가할수록, 공간의 부피가 기하급수적으로 증가하고 데이터는 공간에 희소해져 데이터의 분포 분석이나 모델 추정에 필요한 샘플 데이터 개수가 기하급수적으로 증가.
<img src="https://user-images.githubusercontent.com/18206655/89172882-5385c680-d5be-11ea-87df-eb9b3419a1e6.jpg" width=90%></img>

#### Overfitting and Underfitting   
* Overfitting : 정확한 결과를 얻기 위해 학습데이터의 잡음까지 학습하여 훈련데이터에 최적화되어 있지만 일반화하지 못한 모델.   
Overfitting 하지 못하도록 Dropout(0.5) 함수를 사용한다. 랜덤하게 50% 뉴런을 종료시킨다.
* Underfitting : 학습데이터가 부족하거나 학습이 제대로 이루어지지 않아 훈련집합의 모델이 너무 간단하게 하여 정확도가 낮은 모델   

### Model Test(Training, Validation, Test)
1. 데이터분할: 전체 데이터를 학습데이터, 검증데이터, 테스트 데이터로 나눔   
2. 모델학습: 학습데이터를 사용하여 각 모델을 학습함.   
3. 모델선택: 검증데이터를 사용하여 각 모델의 성능을 비교하고 모형 선택   
4. 최종 성능 지표 도출: 테스트 데이터를 사용하여 검증 데이터로 도출한 최종 모델의 성능 지표를 계산   
5. Training Data: 모형 f를 추정   
6. Validation Data: Overfitting, Underfitting check(f의 적합성 검증)   
7. Test Data: 실제 현장에서 사용하는 데이터, 최종 성능평가 hyperparameter 선택   

#### K-Fold Cross Validation   
데이터가 적은 경우 활용하여 방법으로 데이터를 K개로 나눈 뒤, 그 중 하나를 검증집합, 나머지를 학습집합으로 분류. 이 과정을 K번 반복하고 K개의 성능 지표를 평균하여 모델 적합성을 평가   

#### LOOCV(Leave-One-Out Cross Validation)   
100개 이하의 아주 작은 데이터인 경우 데이터 수만큼의 모델을 만드는데 각 모델은 하나의 샘플만 제외하고 모델을 만들고 제외한 샘플로 성능 계산, 도출된 n개의 성능 지표의 평균을 최종 성능 지표를 도출하는 방법.(의료정보 분석등..)   
![loocv](https://user-images.githubusercontent.com/18206655/89173785-c0e62700-d5bf-11ea-94bd-6ef53211de6f.jpg)   

### Neural Network   
뇌의 학습 방법을 수학적으로 모델링하는 기계학습 알고리즘으로써, 시냅스의 결합으로 네트워크를 형성한 신경세포가 학습을 통해 시냅스의 세기를 변화시켜 문제를 해결하는 모델.   
아래 그림은 Neural Network에서 하나의 component를 보여주고 있다.   
<img src="https://user-images.githubusercontent.com/18206655/89284486-d1fb6a80-d689-11ea-8b1e-b7160086a951.jpg" width=70%></img>      

### Deep Learning   
입력과 출력 사이에 있는 인공 뉴런들을 여러 개로 층층이 쌓고 연결한 인공신경망 기법을 다루는 연구   
<img src="https://user-images.githubusercontent.com/18206655/89174065-3651f780-d5c0-11ea-86ea-019e0bc9cfc0.jpg" width=70%></img>   
hidden layer의 개수를 dense라고 한다. dense의 개수는 2^n개로 설정한다.   
머신러닝 결과는 아래와 같이 3가지 분류로 나타난다.   
1. 값 회귀(regression) -> output을 그냥 받는다.      
2. 이진분류(true or false) -> output에 sigmoid을 적용한다.   
3. 다중분류(Ex. 개, 고양이, 호랑이, 사자등을 구분하는 방법등) -> output에 softmax을 적용한다.   
그렇다면 우리가 자주듣는 자율 주행 자동차는 output으로 어떠한 값이 나올까? 자율 주행 자동차는 2가지 output만 필요하다. accelator와 break를 이용한 속도 제어와 direction에 대한 output만이 필요하다.(생각외로 output은 간단한다?!)    

### ML(Machine Learning) vs. DL(Deep Learning)     
* ML: Input -> Feature extraction(executed by human) ->  Classification -> Output
* DL: Input -> Feature extraction + Classification -> Output   
ML은 hidden layer의 층수가 1층이고, DL은 hidden layer의 층수가 n층으로 설정한다.   
ImageNet에서 AlexNet이 나오기 전에 비전 알고리즘은 ML을 사용했다. 하지만 AlexNet부터 DL이 사용되었다.

#### Backpropagation   
Supervised Learning 기반에서 신경망을 학습시키는 방법으로 최적화의 계산 방향이 출력층에서 시작하여 앞으로 진행하는 방법.   
<img src="https://user-images.githubusercontent.com/18206655/89175110-0a377600-d5c2-11ea-80b7-f0a7bd4850cb.jpg" width=70%></img>   
[Reference link]: https://sebastianraschka.com/faq/docs/visual-backpropagation.html   

#### Gradient descent      
머신러닝을 식 하나로 표현하자만 y=wx라고 간단하게 표현할 수 있다. w는 weight(가중치)이고, x의 입력 값은 행렬이다. 그러므로 y값을 x로 나눌 수 없다. 식을 변형해서 0=wx-y와 같은 형태를 사용해서 비용함수가 0이 되는 값을 구한다. 비용을 0으로 만들어 주는게 gradient descent이다. gradient descent는 MSE(Mean Square Error)를 사용해서 비용함수가 0이 되도록 predictive 선형그래프를 real 선형그래프로 이동시키는 과정을 학습이라고 할 수 있다. 엄밀히 gradient descent를 다시 정의하자만 오차의 최소값 위치를 찾기 위해 Cost Function의 gradient 반대 방향으로 정의한 step size를 가지고 조금씩 움직여 가면서 최적의 parameter(weight)를 찾는 최적화 알고리즘이라고 할 수 있다.     
<img src="https://user-images.githubusercontent.com/18206655/89286574-6e733c00-d68d-11ea-9839-c5b4eac93b65.jpg" width=70%></img>   

### Machine Learning의 문제점   
1. Underfitting
2. Slow
3. Overfitting   

Neural Network의 학습방법은 위에서 설명한 Backpropagation을 사용한다. 예를 들어, 아파트를 구매하기 위해서 아파트의 실 거래 값을 알아보았다. 알아본 결과 아파트의 값이 5억인데 인공지능을 연구하는 학자는 호기심에 본인이 설계한 Neural Network에 아파트의 가격을 예측하기 위해서 학습을 시켜봤다. 그런데 큰일나게도 Neural Network는 아파트의 가격을 3억이라고 예측했다. 그러므로 이 문제를 해결하기 위해 인공지능 학자는 Backpropagation을 사용해서 현재 내가 틀린정도를 미분(기울기)해서 뒤로 전달했다. 그런데 backpropagation을 하다보니까 vanishing gradient 현상이 발생하게 된다. vanishing gradient는 레이어가 깊을 수록 업데이트가 사라져가고 그래서 fitting이 잘 안되는 것을 나타낸다(Underfitting). 그렇다면 왜 이런 문제가 발생한 것인가? 학자는 고민해 보았다. 원인을 찾아보니 activation 함수로 sigmoid를 사용하고 있었는데 sigmoid가 -0.5 < x < 0.5에서는 미분이 가능한데 x <= -0.5 와 x >= 0.5 범위에서 미분하면 기울기가 0이 된다. 이 값을 backpropagation하면 중간 node가 0하고 곱해져서 노드가 꺼지게 된다. 결국 backprogation을 할수록 정보가 사라지게 된다!!. 그래서 인공지능 학자들은 이 문제를 해결하기 위해 ReLU(Rectified Linear Units)를 개발하게 되었다. ReLU는 양의 구간에서 전부 미분값(1)이 있다.       
<img src="https://user-images.githubusercontent.com/18206655/89286188-b9408400-d68c-11ea-94fa-8765c655e064.jpg" width=80%></img>   
Figure Sigmoid.     

<img src="https://user-images.githubusercontent.com/18206655/89286209-c1002880-d68c-11ea-9911-7c2d00f6144d.jpg" width=70%></img>   
Figure ReLu.   

GD(Gradient Descent)는 현재 가진 weight setting 자리에서 내가 가진 데이터를 다 넣어서 전체 error를 계산한다. 계산된 값을 미분하면 error을 줄이는 방향을 알 수 있다.(내자리의 기울기x반대방향) 하지만 실세계에서 트레이닝 데이터는 몇억건을 넘어간다. 그러면 1 step 이동하기 위해서 매번 몇억건을 넣는 것은 너무나도 비효율적이다. 그래서 GD에서 개선된 optimizer인 SGD(Stochastic Gradient Decent)가 나왔다. GD가 전부다 읽고 나서 최적의 1 step을 가는 것에 비해, SGD는 mini-batch마다 일단 1 step을 간다.     
<img src="https://user-images.githubusercontent.com/18206655/89286979-14bf4180-d68e-11ea-94f1-c43eb8748299.jpg" width="60%" height="60%"></img>   
현재 가장 많이 사용하는 optimizer는 Adam으로 90% 정도의 정확도를 가진다. 나머지 10% case에서는 다른 optimizer를 사용해봐야 한다. 즉, 잘 모르겠으면 Adam을 사용하자!   

### CNN vs. RNN or LSTM   
이미지, 영상등 snapshot 데이터는 CNN을 사용하고 음성, 언어, 주식가격처럼 sequence가 있는 데이터는 RNN or LSTM을 사용한다. CNN은 convolution이라는 특정 패턴이 있는지 박스로 훑으며 마킹하는 과정을 통해서 여러가지 패턴에 대해서 확인한다. Convolution 결과값으로 숫자가 나오는데 그 값을 activation function인 ReLU에 넣어 나온 결과 값으로 이미지 지도를 새로 그린다. 아래 링크는 CNN에 대한 매우 상세한 설명을 담은 글이다.   
[CNN Reference Link]: https://cezannec.github.io/Convolutional_Neural_Networks/   
RNN은 내부적으로 sigmoid를 사용하고 있어서 backpropagation중에 vanishing gradient가 발생한다. 그래서 RNN은 학문적으로만 사용하고, 실제 산업계에서는 LSTM을 사용한다. LSTM은 RNN처럼 backpropagation할 때 데이터가 사라지지 않도록 메모리에 데이터를 저장한다.   












