```python
#!pip install sklearn
```


```python
import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
```


```python
class MF(object):
    """docstring for CF"""
    def __init__(self, Y_data, K, lam = 0.1, Xinit = None, Winit = None, 
            learning_rate = 0.5, max_iter = 1000, print_every = 100, user_based = 1):
        self.Y_raw_data = Y_data
        self.K = K
        # regularization parameter
        self.lam = lam
        # learning rate for gradient descent
        self.learning_rate = learning_rate
        # maximum number of iterations
        self.max_iter = max_iter
        # print results after print_every iterations
        self.print_every = print_every
        # user-based or item-based
        self.user_based = user_based
        # number of users, items, and ratings. Remember to add 1 since id starts from 0
        self.n_users = int(np.max(Y_data[:, 0])) + 1 
        self.n_items = int(np.max(Y_data[:, 1])) + 1
        self.n_ratings = Y_data.shape[0]
        
        if Xinit is None: # new
            self.X = np.random.randn(self.n_items, K)
        else: # or from saved data
            self.X = Xinit 
        
        if Winit is None: 
            self.W = np.random.randn(K, self.n_users)
        else: # from daved data
            self.W = Winit
            
        # normalized data, update later in normalized_Y function
        self.Y_data_n = self.Y_raw_data.copy()


    def normalize_Y(self):
        if self.user_based:
            user_col = 0
            item_col = 1
            n_objects = self.n_users

        # if we want to normalize based on item, just switch first two columns of data
        else: # item bas
            user_col = 1
            item_col = 0 
            n_objects = self.n_items

        users = self.Y_raw_data[:, user_col] 
        self.mu = np.zeros((n_objects,))
        for n in range(n_objects):
            # row indices of rating done by user n
            # since indices need to be integers, we need to convert
            ids = np.where(users == n)[0].astype(np.int32)
            # indices of all ratings associated with user n
            item_ids = self.Y_data_n[ids, item_col] 
            # and the corresponding ratings 
            ratings = self.Y_data_n[ids, 2]
            # take mean
            m = np.mean(ratings) 
            if np.isnan(m):
                m = 0 # to avoid empty array and nan value
            self.mu[n] = m
            # normalize
            self.Y_data_n[ids, 2] = ratings - self.mu[n]
            
    #Tính giá trị hàm mất mát
    def loss(self):
        L = 0 
        for i in range(self.n_ratings):
            # user, item, rating
            n, m, rate = int(self.Y_data_n[i, 0]), int(self.Y_data_n[i, 1]), self.Y_data_n[i, 2]
            L += 0.5*(rate - self.X[m, :].dot(self.W[:, n]))**2
        
        # take average
        L /= self.n_ratings
        # regularization, don't ever forget this 
        L += 0.5*self.lam*(np.linalg.norm(self.X, 'fro') + np.linalg.norm(self.W, 'fro'))
        return L 
    
    #Xác định các items được đánh giá bởi 1 user, và users đã đánh giá 1 item và các ratings tương ứng:
    def get_items_rated_by_user(self, user_id):
        """
        get all items which are rated by user user_id, and the corresponding ratings
        """
        ids = np.where(self.Y_data_n[:,0] == user_id)[0] 
        item_ids = self.Y_data_n[ids, 1].astype(np.int32) # indices need to be integers
        ratings = self.Y_data_n[ids, 2]
        return (item_ids, ratings)
        
        
    def get_users_who_rate_item(self, item_id):
        """
        get all users who rated item item_id and get the corresponding ratings
        """
        ids = np.where(self.Y_data_n[:,1] == item_id)[0] 
        user_ids = self.Y_data_n[ids, 0].astype(np.int32)
        ratings = self.Y_data_n[ids, 2]
        return (user_ids, ratings)
    
    #Cap nhat X, W
    def updateX(self):
        for m in range(self.n_items):
            user_ids, ratings = self.get_users_who_rate_item(m)
            Wm = self.W[:, user_ids]
            # gradient
            grad_xm = -(ratings - self.X[m, :].dot(Wm)).dot(Wm.T)/self.n_ratings + \
                                               self.lam*self.X[m, :]
            self.X[m, :] -= self.learning_rate*grad_xm.reshape((self.K,))
    
    def updateW(self):
        for n in range(self.n_users):
            item_ids, ratings = self.get_items_rated_by_user(n)
            Xn = self.X[item_ids, :]
            # gradient
            grad_wn = -Xn.T.dot(ratings - Xn.dot(self.W[:, n]))/self.n_ratings + \
                        self.lam*self.W[:, n]
            self.W[:, n] -= self.learning_rate*grad_wn.reshape((self.K,))
    
    #Phan thuat toan chinh
    def fit(self):
        self.normalize_Y()
        for it in range(self.max_iter):
            self.updateX()
            self.updateW()
            if (it + 1) % self.print_every == 0:
                rmse_train = self.evaluate_RMSE(self.Y_raw_data)
                print ('iter =', it + 1, ', loss =', self.loss(), ', RMSE train =', rmse_train)
    #Du doan
    def pred(self, u, i):
        """ 
        predict the rating of user u for item i 
        if you need the un
        """
        u = int(u)
        i = int(i)
        if self.user_based:
            bias = self.mu[u]
        else: 
            bias = self.mu[i]
        pred = self.X[i, :].dot(self.W[:, u]) + bias 
        # truncate if results are out of range [0, 5]
        if pred < 0:
            return 0 
        if pred > 5: 
            return 5 
        return pred 
        
    
    def pred_for_user(self, user_id):
        """
        predict ratings one user give all unrated items
        """
        ids = np.where(self.Y_data_n[:, 0] == user_id)[0]
        items_rated_by_u = self.Y_data_n[ids, 1].tolist()              
        
        y_pred = self.X.dot(self.W[:, user_id]) + self.mu[user_id]
        predicted_ratings= []
        for i in range(self.n_items):
            if i not in items_rated_by_u:
                predicted_ratings.append((i, y_pred[i]))
        
        return predicted_ratings
    
    #Du doan ket qua bang cach do Root Mean Square Error
    def evaluate_RMSE(self, rate_test):
        n_tests = rate_test.shape[0]
        SE = 0 # squared error
        for n in range(n_tests):
            pred = self.pred(rate_test[n, 0], rate_test[n, 1])
            SE += (pred - rate_test[n, 2])**2 

        RMSE = np.sqrt(SE/n_tests)
        return RMSE
    #Du doan ket qua bang cach do Root Mean Square Error
    def evaluate_RMSE(self, rate_test):
        n_tests = rate_test.shape[0]
        SE = 0 # squared error
        for n in range(n_tests):
            pred = self.pred(rate_test[n, 0], rate_test[n, 1])
            SE += (pred - rate_test[n, 2])**2 

        RMSE = np.sqrt(SE/n_tests)
        return RMSE
    def evaluate_MSE(self, rate_test):
        n_tests = rate_test.shape[0]
        SE = 0  # squared error
        for n in range(n_tests):
            pred = self.pred(rate_test[n, 0], rate_test[n, 1])
            SE += (pred - rate_test[n, 2]) ** 2

        MSE = (SE / n_tests)
        return MSE

    def evaluate_MAE(self, rate_test):
        n_tests = rate_test.shape[0]
        SE = 0  # squared error
        for n in range(n_tests):
            pred = self.pred(rate_test[n, 0], rate_test[n, 1])
            SE += abs(pred - rate_test[n, 2])

        MAE = SE / n_tests
        return MAE
    
```


```python
#Ap dung tren MovieLens 100k

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings_base = pd.read_csv('ml-100k/ub.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ub.test', sep='\t', names=r_cols, encoding='latin-1')
rate_train = ratings_base[["user_id", "movie_id", "rating", "unix_timestamp"]].values
rate_test = ratings_test[["user_id", "movie_id", "rating", "unix_timestamp"]].values
#rate_train = ratings_base.as_matrix()
#rate_test = ratings_test.as_matrix()

# indices start from 0
rate_train[:, :2] -= 1
rate_test[:, :2] -= 1
```


```python
rs = MF(rate_train, K = 10, lam = .1, print_every = 10, 
    learning_rate = 0.75, max_iter = 100, user_based = 1)
rs.fit()
# evaluate on test data
RMSE = rs.evaluate_RMSE(rate_test)
print ('\nUser-based MF, RMSE =', RMSE)
```

    iter = 10 , loss = 5.635816602261956 , RMSE train = 1.2106979092550387
    iter = 20 , loss = 2.6305929237820287 , RMSE train = 1.0385371478956549
    iter = 30 , loss = 1.3391975486598595 , RMSE train = 1.0296535899087977
    iter = 40 , loss = 0.7509143371701861 , RMSE train = 1.0292375048727784
    iter = 50 , loss = 0.48132756714409813 , RMSE train = 1.029215792872691
    iter = 60 , loss = 0.35771065612752295 , RMSE train = 1.0292144136373753
    iter = 70 , loss = 0.301023028918587 , RMSE train = 1.0292143228838426
    iter = 80 , loss = 0.2750272092323949 , RMSE train = 1.0292143248905197
    iter = 90 , loss = 0.2631060091271385 , RMSE train = 1.029214328827978
    iter = 100 , loss = 0.25763916172379775 , RMSE train = 1.0292143303547698
    
    User-based MF, RMSE = 1.060379900830674
    


```python
#Kết quả nếu sư dụng cách chuẩn hoá dựa trên user:
rs = MF(rate_train, K = 10, lam = .1, print_every = 10, 
    learning_rate = 0.75, max_iter = 100, user_based = 1)
rs.fit()
# evaluate on test data
RMSE = rs.evaluate_RMSE(rate_test)
MAE = rs.evaluate_MAE(rate_test)
MSE = rs.evaluate_MSE(rate_test)
print ('\nUser-based MF, RMSE =', RMSE)
print ('\nUser-based MF, RMSE =', MAE)
print ('\nUser-based MF, RMSE =', MSE)
```

    iter = 10 , loss = 5.7139095827875295 , RMSE train = 1.2151242815632681
    iter = 20 , loss = 2.6631483762113137 , RMSE train = 1.038672446487276
    iter = 30 , loss = 1.3540732586526592 , RMSE train = 1.0296189432198575
    iter = 40 , loss = 0.7577540748675247 , RMSE train = 1.0292273697863303
    iter = 50 , loss = 0.4844699433798214 , RMSE train = 1.0292135288039774
    iter = 60 , loss = 0.3591533188517295 , RMSE train = 1.029213930222031
    iter = 70 , loss = 0.3016851125093995 , RMSE train = 1.0292142205999215
    iter = 80 , loss = 0.2753310083265193 , RMSE train = 1.029214303289058
    iter = 90 , loss = 0.2632453972815276 , RMSE train = 1.029214324267632
    iter = 100 , loss = 0.25770311305422267 , RMSE train = 1.029214329392071
    
    User-based MF, RMSE = 1.0603799087986703
    
    User-based MF, RMSE = 0.8485804991055672
    
    User-based MF, RMSE = 1.1244055509838764
    


```python
#Nếu chuẩn hoá dựa trên item:
rs = MF(rate_train, K = 10, lam = .1, print_every = 10,
        learning_rate = 0.75, max_iter = 100, user_based = 0)
rs.fit()
# evaluate on test data
RMSE = rs.evaluate_RMSE(rate_test)
MAE = rs.evaluate_MAE(rate_test)
MSE = rs.evaluate_MSE(rate_test)
print ('\nItem-based MF, RMSE =', RMSE)
print ('\nItem-based MF, MAE =', MAE)
print ('\nItem-based MF, MSE =', MSE)
```

    c:\users\doantu\anaconda3\envs\myenv\lib\site-packages\numpy\core\fromnumeric.py:3335: RuntimeWarning: Mean of empty slice.
      out=out, **kwargs)
    c:\users\doantu\anaconda3\envs\myenv\lib\site-packages\numpy\core\_methods.py:161: RuntimeWarning: invalid value encountered in double_scalars
      ret = ret.dtype.type(ret / rcount)
    

    iter = 10 , loss = 5.625108888894107 , RMSE train = 1.1825439710808252
    iter = 20 , loss = 2.614534659029628 , RMSE train = 1.0058211424403363
    iter = 30 , loss = 1.322601421403159 , RMSE train = 0.9966008682473602
    iter = 40 , loss = 0.7341252769205351 , RMSE train = 0.9961962091058988
    iter = 50 , loss = 0.464445800014513 , RMSE train = 0.9961805525363127
    iter = 60 , loss = 0.3407846335770525 , RMSE train = 0.9961806002828201
    iter = 70 , loss = 0.28407623567189455 , RMSE train = 0.9961808147494139
    iter = 80 , loss = 0.25807075339768937 , RMSE train = 0.9961808808205371
    iter = 90 , loss = 0.246145076762944 , RMSE train = 0.9961808981516024
    iter = 100 , loss = 0.24067615943435772 , RMSE train = 0.9961809024760411
    
    Item-based MF, RMSE = 1.049804747929377
    
    Item-based MF, MAE = 0.8413935138401717
    
    Item-based MF, MSE = 1.1020900087750631
    


```python
#Chúng ta cùng làm thêm một thí nghiệm nữa khi không sử dụng regularization, tức lam = 0:
rs = MF(rate_train, K = 2, lam = 0, print_every = 10,
        learning_rate = 1, max_iter = 100, user_based = 0)
rs.fit()
# evaluate on test data
RMSE = rs.evaluate_RMSE(rate_test)
MAE = rs.evaluate_MAE(rate_test)
MSE = rs.evaluate_MSE(rate_test)
print ('\nItem-based MF, RMSE =', RMSE)
print ('\nItem-based MF, MAE =', MAE)
print ('\nItem-based MF, MSE =', MSE)
```

    iter = 10 , loss = 1.14636551607477 , RMSE train = 1.4744308812240106
    iter = 20 , loss = 1.0832465174465291 , RMSE train = 1.454798993635731
    iter = 30 , loss = 1.0269823409790022 , RMSE train = 1.4363374343326094
    iter = 40 , loss = 0.9765890016676828 , RMSE train = 1.419031087271316
    iter = 50 , loss = 0.9312551600536436 , RMSE train = 1.4027088316407368
    iter = 60 , loss = 0.8903067351332513 , RMSE train = 1.3872761030555445
    iter = 70 , loss = 0.8531797417537539 , RMSE train = 1.3727112709285183
    iter = 80 , loss = 0.8193992304261025 , RMSE train = 1.3589508982398453
    iter = 90 , loss = 0.7885628058360107 , RMSE train = 1.3458941796594972
    iter = 100 , loss = 0.7603276165382622 , RMSE train = 1.3335466112740617
    
    Item-based MF, RMSE = 1.4259910698891152
    
    Item-based MF, MAE = 1.1113358650872347
    
    Item-based MF, MSE = 2.0334505314035036
    


```python
#loading user profile from u.user to user_info dataframe
userCols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
users_info = pd.read_csv('ml-100k/u.user', sep='|', names=userCols)
users_info.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>age</th>
      <th>gender</th>
      <th>occupation</th>
      <th>zip_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>24</td>
      <td>M</td>
      <td>technician</td>
      <td>85711</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>53</td>
      <td>F</td>
      <td>other</td>
      <td>94043</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>23</td>
      <td>M</td>
      <td>writer</td>
      <td>32067</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>24</td>
      <td>M</td>
      <td>technician</td>
      <td>43537</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>33</td>
      <td>F</td>
      <td>other</td>
      <td>15213</td>
    </tr>
  </tbody>
</table>
</div>




```python
#loading movie profile from u.item to movie_info dataframe
with open('./ml-100k/u.item', encoding = "ISO-8859-1") as content:
    mCols =     ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'imdb_url']
    genres = ['unknown', 'action', 'adventure', 'animation', 'children', 'comedy', 'crime', 'documentary', 'drama', 'fantasy', 
          'film-noir',  'horror', 'musical', 'mystery', 'romance', 'sci-fi', 'thriller', 'war', 'western']
    mColsG = mCols + genres
    movies_info = pd.DataFrame(columns=mColsG)
    i = 0
    for x in content:
        x = x.split("|")
        x[-1] = x[-1][:-1]
        if x[1][-1] == ' ':
            x[1] = x[1][:-1]
        movies_info.loc[i] = [word if word!='' else "empty" for word in x]
        i = i + 1
movies_info['movie_id'] = movies_info['movie_id'].astype('int64')
movies_info[genres] = movies_info[genres].astype('int64')
movies_info.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movie_id</th>
      <th>movie_title</th>
      <th>release_date</th>
      <th>video_release_date</th>
      <th>imdb_url</th>
      <th>unknown</th>
      <th>action</th>
      <th>adventure</th>
      <th>animation</th>
      <th>children</th>
      <th>...</th>
      <th>fantasy</th>
      <th>film-noir</th>
      <th>horror</th>
      <th>musical</th>
      <th>mystery</th>
      <th>romance</th>
      <th>sci-fi</th>
      <th>thriller</th>
      <th>war</th>
      <th>western</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>01-Jan-1995</td>
      <td>empty</td>
      <td>http://us.imdb.com/M/title-exact?Toy%20Story%2...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>GoldenEye (1995)</td>
      <td>01-Jan-1995</td>
      <td>empty</td>
      <td>http://us.imdb.com/M/title-exact?GoldenEye%20(...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Four Rooms (1995)</td>
      <td>01-Jan-1995</td>
      <td>empty</td>
      <td>http://us.imdb.com/M/title-exact?Four%20Rooms%...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Get Shorty (1995)</td>
      <td>01-Jan-1995</td>
      <td>empty</td>
      <td>http://us.imdb.com/M/title-exact?Get%20Shorty%...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Copycat (1995)</td>
      <td>01-Jan-1995</td>
      <td>empty</td>
      <td>http://us.imdb.com/M/title-exact?Copycat%20(1995)</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
#remove video_release_date since column values are empty
if 'video_release_date' in movies_info.columns:
    movies_info = movies_info.drop('video_release_date', axis=1)
```


```python
#creating movie profile with '|' separated genres
movies_genres = movies_info.copy()
named = []
for i in range(0,len(movies_genres)):
    genre = ""
    for column in movies_genres.columns[5:]: 
        if (movies_genres.iloc[i][column] == 1):
            genre = genre + column + '|'
    genre = genre[:-1]
    named.append(genre)
movies_genres['genre_names'] = named
movies_genres['genre_names'] = movies_genres['genre_names'].astype('str')
movies_genres = movies_genres.drop(movies_genres.columns[list(range(4,23))], axis=1) 
movies_genres.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movie_id</th>
      <th>movie_title</th>
      <th>release_date</th>
      <th>imdb_url</th>
      <th>genre_names</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>01-Jan-1995</td>
      <td>http://us.imdb.com/M/title-exact?Toy%20Story%2...</td>
      <td>animation|children|comedy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>GoldenEye (1995)</td>
      <td>01-Jan-1995</td>
      <td>http://us.imdb.com/M/title-exact?GoldenEye%20(...</td>
      <td>action|adventure|thriller</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Four Rooms (1995)</td>
      <td>01-Jan-1995</td>
      <td>http://us.imdb.com/M/title-exact?Four%20Rooms%...</td>
      <td>thriller</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Get Shorty (1995)</td>
      <td>01-Jan-1995</td>
      <td>http://us.imdb.com/M/title-exact?Get%20Shorty%...</td>
      <td>action|comedy|drama</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Copycat (1995)</td>
      <td>01-Jan-1995</td>
      <td>http://us.imdb.com/M/title-exact?Copycat%20(1995)</td>
      <td>crime|drama|thriller</td>
    </tr>
  </tbody>
</table>
</div>




```python
#loading user-movie profile from u.data to user_movie_info dataframe
umCols = ['user_id', 'movie_id', 'rating', 'timestamp']
user_movie_info = pd.read_csv('ml-100k/u.data', sep='\t', names=umCols)
user_movie_info.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>movie_id</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>196</td>
      <td>242</td>
      <td>3</td>
      <td>881250949</td>
    </tr>
    <tr>
      <th>1</th>
      <td>186</td>
      <td>302</td>
      <td>3</td>
      <td>891717742</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22</td>
      <td>377</td>
      <td>1</td>
      <td>878887116</td>
    </tr>
    <tr>
      <th>3</th>
      <td>244</td>
      <td>51</td>
      <td>2</td>
      <td>880606923</td>
    </tr>
    <tr>
      <th>4</th>
      <td>166</td>
      <td>346</td>
      <td>1</td>
      <td>886397596</td>
    </tr>
  </tbody>
</table>
</div>




```python
#creating a new multi_movie dataframe by splitting multi-genre movies into multiple 'same' movies with single genre
new = ([(row['movie_id'], row['movie_title'], row['genre_names'].split('|'))              
                    for _, row in movies_genres.iterrows()])
movCol = ['movie_id', 'movie_title', 'genre']
multi_movie = pd.DataFrame(columns=movCol)
i = 0
for num1 in range(0,len(new)-1):
    for num2 in range(0,len(new[num1][2])):
        multi_movie.loc[i, 'movie_id'] = new[num1][0]
        multi_movie.loc[i, 'movie_title'] = new[num1][1]
        multi_movie.loc[i, 'genre'] = new[num1][2][num2]
        i = i + 1
multi_movie.to_csv('./data01/movie_genre.txt', columns= ['movie_title', 'genre'] ,sep='\t',index=False,header=False,float_format='%.0f')
multi_movie.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movie_id</th>
      <th>movie_title</th>
      <th>genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>animation</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>children</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>comedy</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>GoldenEye (1995)</td>
      <td>action</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>GoldenEye (1995)</td>
      <td>adventure</td>
    </tr>
  </tbody>
</table>
</div>




```python
#forming multi_user_movie dataframe after splitting multi-genre movies into multiple 'same' movies with single genre
multi_user_movie = user_movie_info.merge(movies_genres, left_on='movie_id', right_on='movie_id', how='inner')
multi_user_movie.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>movie_id</th>
      <th>rating</th>
      <th>timestamp</th>
      <th>movie_title</th>
      <th>release_date</th>
      <th>imdb_url</th>
      <th>genre_names</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>196</td>
      <td>242</td>
      <td>3</td>
      <td>881250949</td>
      <td>Kolya (1996)</td>
      <td>24-Jan-1997</td>
      <td>http://us.imdb.com/M/title-exact?Kolya%20(1996)</td>
      <td>comedy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>63</td>
      <td>242</td>
      <td>3</td>
      <td>875747190</td>
      <td>Kolya (1996)</td>
      <td>24-Jan-1997</td>
      <td>http://us.imdb.com/M/title-exact?Kolya%20(1996)</td>
      <td>comedy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>226</td>
      <td>242</td>
      <td>5</td>
      <td>883888671</td>
      <td>Kolya (1996)</td>
      <td>24-Jan-1997</td>
      <td>http://us.imdb.com/M/title-exact?Kolya%20(1996)</td>
      <td>comedy</td>
    </tr>
    <tr>
      <th>3</th>
      <td>154</td>
      <td>242</td>
      <td>3</td>
      <td>879138235</td>
      <td>Kolya (1996)</td>
      <td>24-Jan-1997</td>
      <td>http://us.imdb.com/M/title-exact?Kolya%20(1996)</td>
      <td>comedy</td>
    </tr>
    <tr>
      <th>4</th>
      <td>306</td>
      <td>242</td>
      <td>5</td>
      <td>876503793</td>
      <td>Kolya (1996)</td>
      <td>24-Jan-1997</td>
      <td>http://us.imdb.com/M/title-exact?Kolya%20(1996)</td>
      <td>comedy</td>
    </tr>
  </tbody>
</table>
</div>




```python
#creating movie_title - rating dataframe
movie_rating = multi_user_movie.copy()
movie_rating = movie_rating.drop(movie_rating.columns[0:2], axis=1)
movie_rating = movie_rating.drop(movie_rating.columns[1:2], axis=1)
movie_rating = movie_rating.drop(movie_rating.columns[2:], axis=1)
movie_rating.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
      <th>movie_title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>Kolya (1996)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>Kolya (1996)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>Kolya (1996)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Kolya (1996)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Kolya (1996)</td>
    </tr>
  </tbody>
</table>
</div>




```python
#creating movie: [all the ratings of that movie] dictionary
movie_rating_map = {}
#iterate over all movies
for keyval in movie_rating['movie_title']:
    rat = []
    #iterate over all movies rated by a user
    if keyval in movie_rating_map.keys():
        movie_rating_map[keyval].append(movie_rating.loc[movie_rating['movie_title'] == keyval, 'rating'].iloc[len(movie_rating_map[keyval])])
    else:
        rat.append(movie_rating.loc[movie_rating['movie_title'] == keyval, 'rating'].iloc[0])
        movie_rating_map[keyval] = rat
#movie_rating_map
```


```python
#creating movie: average rating dictionary
avg_rating_map = {}
for k,v in movie_rating_map.items():
    avg_rating_map[k] = sum(v)/ float(len(v))
#avg_rating_map
```


```python
#converting the above dictionary to dataframe
avg_rating = pd.DataFrame(columns=['movie_title', 'rating'])
index = 0
for k,v in avg_rating_map.items():
    index = index + 1
    avg_rating.loc[index, 'movie_title'] = k
    avg_rating.loc[index, 'rating'] = v
#avg_rating_map
```


```python
#saving the above dataframe to file
avg_rating.to_csv('./data01/movie_avg_rating.txt', columns= ['movie_title', 'rating'] ,sep='\t',index=False,header=False,float_format='%.0f',encoding='latin-1')
```


```python
#removing columns with unknown values
multi_user_movie = multi_user_movie[multi_user_movie['release_date'] != 'empty']
multi_user_movie.to_csv('./data01/jlt/multi_user_movie_full.txt', sep='\t',index=False,header=False,float_format='%.0f')
multi_user_movie.to_csv('./data01/jlt/multi_user_movie.txt', columns= ['user_id', 'movie_title'] ,sep='\t',index=False,header=False,float_format='%.0f')
ncols = ['user_id', 'movie_id', 'rating', 'timestamp', 'movie_title','release_date', 'imdb_url', 'genre_names']
multi_user_movie = pd.read_csv('./data01/jlt/multi_user_movie_full.txt', sep='\t', names=ncols, encoding='latin-1')
```


```python
#
movie_without_year = movies_genres.copy()
movie_without_year['movie_title'] = movie_without_year['movie_title'].map(lambda x: str(x)[:-7])
movie_without_year.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movie_id</th>
      <th>movie_title</th>
      <th>release_date</th>
      <th>imdb_url</th>
      <th>genre_names</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story</td>
      <td>01-Jan-1995</td>
      <td>http://us.imdb.com/M/title-exact?Toy%20Story%2...</td>
      <td>animation|children|comedy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>GoldenEye</td>
      <td>01-Jan-1995</td>
      <td>http://us.imdb.com/M/title-exact?GoldenEye%20(...</td>
      <td>action|adventure|thriller</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Four Rooms</td>
      <td>01-Jan-1995</td>
      <td>http://us.imdb.com/M/title-exact?Four%20Rooms%...</td>
      <td>thriller</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Get Shorty</td>
      <td>01-Jan-1995</td>
      <td>http://us.imdb.com/M/title-exact?Get%20Shorty%...</td>
      <td>action|comedy|drama</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Copycat</td>
      <td>01-Jan-1995</td>
      <td>http://us.imdb.com/M/title-exact?Copycat%20(1995)</td>
      <td>crime|drama|thriller</td>
    </tr>
  </tbody>
</table>
</div>




```python
#
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
#returns T x D i.e. term document matrix
tfidf_matrix = tf.fit_transform(movies_info['movie_title'])
```


```python
#
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
```


```python
#
indices = pd.Series(movie_without_year.index, index=movie_without_year['movie_title'])
```


```python
#
def get_similar(title):
    print("The genre of the given movie is:", movie_without_year.loc[movie_without_year['movie_title']==title,'genre_names'].iloc[0])
    idx = indices[title]
    sim_score = list(enumerate(cosine_sim[idx]))
    sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)
    sim_score = sim_score[1:11]#top10
    movie_indices = [i[0] for i in sim_score]
    for num in movie_indices:
        print(movie_without_year['movie_title'].iloc[num],": ",movie_without_year['genre_names'].iloc[num])
```


```python
#
get_similar('Toy Story')
```

    The genre of the given movie is: animation|children|comedy
    Pyromaniac's Love Story, A :  comedy|romance
    Now and Then :  drama
    Show, The :  documentary
    To Have, or Not :  drama
    Story of Xinghua, The :  drama
    Philadelphia Story, The :  comedy|romance
    NeverEnding Story III, The :  children|fantasy
    FairyTale: A True Story :  children|drama|fantasy
    Entertaining Angels: The Dorothy Day Story :  drama
    Police Story 4: Project S (Chao ji ji hua) :  action
    


```python
get_similar('Richard III')
```

    The genre of the given movie is: drama|war
    Looking for Richard :  documentary|drama
    Now and Then :  drama
    Show, The :  documentary
    To Have, or Not :  drama
    NeverEnding Story III, The :  children|fantasy
    Highlander III: The Sorcerer :  action|sci-fi
    Beverly Hills Cop III :  action|comedy
    Land Before Time III: The Time of the Great Giving (19 :  animation|children
    Star Trek III: The Search for Spock :  action|adventure|sci-fi
    Boys on the Side :  comedy|drama
    


```python

```
