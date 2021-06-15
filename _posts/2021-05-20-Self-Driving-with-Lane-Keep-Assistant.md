---
categories: 
   - Projects
tags:
   - 
---
> The real-time lane keep assistant using opencv.


- - - 


## Overall Module Structure
For this project, there will the main module that will connects to all various modules. The reason for having different modules for each different functionalities is that adding or modifying different modules becomes very convenient in this structure for the future upgrade.

![structure](https://raw.githubusercontent.com/yonghoson/yonghoson.github.io/master/images/structure.PNG)

## Step 1 - Finding Lane

Since I used a regular A4 white paper as path, simply applying the color detection will find the path. Thus, the ```thresholding``` function will be defined as below:

```java
def thresholding(img):
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Convert to HSV
    lower_white = np.array([80, 0, 0])            # Define Range of White color in HSV
    upper_white = np.array([255, 160, 255])
    mask_white = cv2.inRange(imgHsv, lower_white, upper_white)
    return mask_white 
    }
```
The above code simply converts the image to HSV color space and then apply a range of color to find the white paper path.

![structure](https://raw.githubusercontent.com/yonghoson/yonghoson.github.io/master/images/imgThres.PNG)

## Step 2 - Bird Eye View (Warping)

```java
    Retrofit retrofit =  new  Retrofit.Builder()  
	    .baseUrl("https://api.github.com/")  
	    .build();
	    
	GitHubService service = retrofit.create(GitHubService.class);
```

생성된 ```GitHubService```로 부터의 ```Call```은 웹 서버로 비동기/동기적인 HTTP 요청을 전송한다.

HTTP 요청을 설명하기 위해 아래의 annotations을 이용할 수 있다.
+ URL parameter 교체
+ query parameter 지원
+ Object를 request body로 변환
+ Multipart로 구성된 body나 파일 업로드

## 2. API 선언
인터페이스 메소드나, 매개 변수에 어노테이션으로 요청을 처리하는 방법에 대한 기술.

### Request Method
모든 메소드는 요청 메소드와 URL에 대한 HTTP 어노테이션이 필요하다.
이에 5개의 내장 어노테이션이 존재하는데, 이는 GET, POST, PUL, DELETE, HEAD이다.

```java
    @GET("users/list")
    // 물론 URL에 매개 변수를 명시할 수 있다.
    @GET("users/list?sort=desc")
```

### URL 조작
아래와 같이 요청하려는 URL을 동적으로 지정 할 수 있다.

```java
    @GET("group/{id}/users")
    Call<List<User>> groupList(@Path("id")  int groupId);
```

기본적으로 위처럼 정적 변수를 입력할 수도 있지만, 쿼리 매개 변수를 사용하여 동적으로 할당하는 것도 가능하다. 

### Request body
HTTP 요청의 본문에 ```@body``` 어노테이션을 이용해 명시할 수 있다.

```java
    @POST("users/new")
    Call<User> createUser(@Body  User user);
```

### Form encoded와 Multipart
Form-encoded data나 Multipart data를 이용해 선언할 수도 있다.

Form-encoded 데이터는 ```FormUrlEncoded```가 현재 메소드에 존재할 때 사용된다.

```java
    @FormUrlEncoded
    @POST("user/edit")
    Call<User> updateUser(@Field("first_name")  String first,  @Field("last_name")  String  last);
```

Multipart 요청은 ```@Multipart```가 현재 메소드에 존재할 때 사용된다,

```java
    @Multipart 
    PUT("user/photo")  
    Call<User> updateUser(@Part("photo")  RequestBody photo,  @Part("description")  RequestBody description);
```

### Header 조작

```java
    @Headers({  
	    "Accept: application/vnd.github.v3.full+json",  
	    "User-Agent: Retrofit-Sample-App"  })  
	@GET("users/{username}")
	Call<User> getUser(@Path("username")  String username);
```
	
동적 헤더는 ```@Headers``` 어노테이션을 이용하여 선언 할 수 있다.

같은 이름을 가진 헤더끼리도 서로 덮어씌우지 않고, 각각 요청에 포함된다.

```@Header```이 null이면 헤더를 생략한다.



### 동기식 vs 비동기식

```call``` 인스턴스는 enqueue를 사용하여 비동기식으로 실행하거나, excute를 사용하여 동기식으로 선택하여 실행할 수 있다.
각 인스턴스는 한 번만 사용할 수 있지만, ```clone()```를 호출하여 새 인스턴스를 생성한 후, 사용할 수 있다.


## 3. 장점

* 빠른 속도, 안정성.
* 쉽게 설치하고, 사용할 수 있다.
* 어노테이션을 사용하여 직관적이고, 가독성이 뛰어나다.
* 플러그인 형태를 취하고 있어, 유지보수에 편하다.


## 4. 단점
* 타 라이브러리에 비해 추가기능이 많은 편은 아니다.
* OkHttp의 상위에서 구현된 라이브러리로, 기본적으로 OkHttp에 의존적이다. <sub>(사실 OkHttp도 매우 좋은 라이브러리로, 그다지 단점은 아님)</sub>

