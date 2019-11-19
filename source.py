
from Tkinter import *
import tkFileDialog as filedialog
from PIL import ImageTk, Image
master = Tk()
master.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
outstring = " "
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
j1=1 
f=open('Results.txt','w')
while j1<=108:#traverse of every test image
    k1=''
    k1=str(j1)
    imgs=k1+".jpg"
    imgs=''.join(imgs)
    img = cv2.imread(master.filename,0)
    img = cv2.bilateralFilter(img,5,75,75)
    print(imgs)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))########  for images in analysis.txt
    img = clahe.apply(img)
    
    _,img_dilation = cv2.threshold(img,120,255,cv2.THRESH_BINARY)
    kernel=np.ones((3,3), np.uint8)####
    img_dilation= cv2.erode(img_dilation, kernel, iterations=1)

    laplacian=cv2.Sobel(img_dilation,cv2.CV_64F,1,0,ksize=-1)

    height, width=laplacian.shape

    count=0
    temp=0
    i=(height//2)-50
    j=100
    x1=0
    x2=0
    set_height=50   
    set_width=215

    while i<height:
        if height-i<set_height:
            break
        j=100
        while j<width:
            if width-j<set_width+100:
                break
            count=0
            j1=j
            i1=i
            count=np.count_nonzero(laplacian[i1:i1+set_height//4,j1:j1+set_width])
            if count<temp//4:
                j+=4 
                continue
            count=np.count_nonzero(laplacian[i1:i1+set_height,j1:j1+set_width])
            if temp<count:
                temp=count
                x1,x2=i,j
            j+=4 
        i+=3   
   
    c_img=img_dilation[x1:x1+set_height, x2:x2+set_width]
     
    img = c_img
    height, width = img.shape 
    org=np.zeros((height,width,1),np.uint8)
    i=0
    j=0
    while i<height:
        while j<width:
            org[i][j]=img[i][j]
            j+=1
        i+=1    
    kernel = np.ones((3,3), np.uint8)####
    img = cv2.dilate(img, kernel, iterations=1)

    img = cv2.erode(img, kernel, iterations=1)

    sobelx = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=-1)
    sobely = cv2.Sobel(img,cv2.CV_8U,0,1,ksize=-1)
    _,sobelx = cv2.threshold(sobelx,128, 255, cv2.THRESH_BINARY)
    _,sobely = cv2.threshold(sobely,128, 255, cv2.THRESH_BINARY)

    i=0
    count=0
    ar=[]
    xar=[]

    i=0 
    j=0
    while i<width:
        j=0
        count=0
        while j<height:
            if sobelx[j][i]==255:
                count+=1
            j+=1
        ar.append(count)
        i+=1


    prev_val_stored=0
    first_val_stored=0
    count=0
    i=0
    j=0
    k=0
    tall=0
    first_point=0
    first_val=0
    prev_val=0
    val=0
    after=49
    last_point=0
    first_point_var=0
    while i<height:
        j=0
        count=0
        while j<width:
            if sobelx[i][j]==255:
                count+=1
            j+=1
        xar.append(count)
        val=count
        if i==0:
            first_val=count
            prev_val=count
            first_point_var=i
        else:
            if val<prev_val:
                temp=prev_val-first_val
                if temp>tall:
                    tall=temp
                    prev_val_stored=prev_val
                    first_val_stored=first_val
                    last_point=i-1
                    first_point=first_point_var
                k=0
            if val>prev_val:
                if k==0:
                    k=1
                    first_val=prev_val
                    first_point_var=i-1
            prev_val=val
        i+=1
    #print "first point->",first_point," start->",first_val_stored,"last point->",last_point," end->",prev_val_stored
    xx=first_point
    yy=last_point

    prev_val_stored=0
    first_val_stored=0
    count=0
    i=0
    j=0
    k=0
    tall=0
    first_point=0
    first_val=0
    prev_val=0
    val=0
    after=49
    last_point=0
    first_point_var=0

    while i<height:
        j=0
        count=0
        while j<width:
            if sobelx[i][j]==255:
                count+=1
            j+=1
        #xar.append(count)
        val=count
        if i==0:
            first_val=count
            prev_val=count
            first_point_var=i
        else:
            if val<prev_val:
                temp=prev_val-first_val
                if temp>tall and first_point_var!=xx and yy!=i-1:
                    tall=temp
                    prev_val_stored=prev_val
                    first_val_stored=first_val
                    last_point=i-1
                    first_point=first_point_var
                k=0
            if val>prev_val:
                if k==0:
                    k=1
                    first_val=prev_val
                    first_point_var=i-1
            prev_val=val
        i+=1
    #print "first point->",first_point," start->",first_val_stored,"last point->",last_point," end->",prev_val_stored

    prev_val_stored=0
    first_val_stored=0
    count=0
    i=height-1
    j=0
    k=1
    tall=0
    first_point1=0
    first_val=0
    prev_val=0
    val=0
    after=49
    last_point1=0
    first_point_var=0
    while i>=0:
        j=0
        count=0
        while j<width:
            if sobelx[i][j]==255:
                count+=1
            j+=1
        #xar.append(count)
        val=count
        if i==height-1:
            first_val=count
            prev_val=count
            first_point_var=i
        else:
            if val<prev_val:
                temp=prev_val-first_val
                if temp>tall:
                    tall=temp
                    prev_val_stored=prev_val
                    first_val_stored=first_val
                    if i!=0:
                        last_point1=i+1
                    else:
                        last_point1=0
                    first_point1=first_point_var
                k=0
            if val>prev_val:
                if k==0:
                    k=1
                    first_val=prev_val
                    first_point_var=i+1
            prev_val=val
        i-=1
    #print "first point->",first_point1," start->",first_val_stored,"last point->",last_point1," end->",prev_val_stored
    xx1=first_point1
    yy1=last_point1

    prev_val_stored=0
    first_val_stored=0
    count=0
    i=height-1
    j=0 
    k=1
    tall=0
    first_point1=0
    first_val=0
    prev_val=0
    val=0
    after=49
    last_point1=0
    first_point_var=0
    while i>=0:
        j=0
        count=0
        while j<width:
            if sobelx[i][j]==255:
                count+=1
            j+=1
        #xar.append(count)
        val=count
        if i==height-1:
            first_val=count
            prev_val=count
            first_point_var=i
        else:
            if val<prev_val:
                temp=prev_val-first_val
                if temp>tall and first_point_var!=xx1 and yy1!=i+1:
                    tall=temp
                    prev_val_stored=prev_val
                    first_val_stored=first_val
                    if i!=0:
                        last_point1=i+1
                    else:
                        last_point1=0
                    first_point1=first_point_var
                k=0
            if val>prev_val:
                if k==0:
                    k=1
                    first_val=prev_val
                    first_point_var=i+1
            prev_val=val
        i-=1
    lp1=last_point1
    lp=last_point
    fp1=first_point1
    fp=first_point
    starting1=0
    ending1=0


    if yy1-yy<last_point1-last_point and ((yy>lp and yy>lp1 and yy1>lp and yy1>lp1)or (yy<lp and yy<lp1 and yy1<lp and yy1<lp1)):
        if (lp+fp)//2<lp-3:
            var=lp-(lp+fp)//2
        else:
            var=3
        if (lp1+fp1)//2<lp1+3:
            var1=(lp1+fp1)//2-lp1
        else:
            var1=3
        starting=(last_point-var)
        ending=(last_point1+var1)
        starting1=(((fp+lp)//2)+fp)//2#(last_point-var)
        ending1=(((fp1+lp1)//2)+fp1)//2#(last_point1+var1
    else:
        if (xx+yy)//2<yy-3:
            var=yy-(xx+yy)//2
        else:
            var=3
        if (xx1+yy1)//2<yy1+3:
            var1=(xx1+yy1)//2-yy1
        else:
            var1=3  
        starting=(yy-var)
        ending=(yy1+var1)
        starting1=(((xx+yy)//2)+xx)//2#(last_point-var)
        ending1=(((xx1+yy1)//2)+xx1)//2#(last_point1+var1


    if starting>=ending:
        j1111=j1111+1
        continue

    if starting<0:
        starting=0
    if ending>=height:
        ending=height-1
    heightss, widthss = img.shape
    c_img=img[starting:ending,0:0+widthss] 
    
    img=img[starting1:ending1,0:0+widthss]
    heightss, widthss = img.shape

    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

    kernel = np.ones((3,3), np.uint8)
    c_img = cv2.dilate(c_img, kernel, iterations=1)
    _,c_img = cv2.threshold(c_img,128, 255, cv2.THRESH_BINARY)

    heightz,widthz= c_img.shape 
    xary=[]
    count1=0
    i=0
    j=0


    while i<widthz:
        j=0
        count1=0
        while j<heightz:
            if c_img[j][i]==0:
                count1+=1
            j+=1
        xary.append(count1)
        i+=1
    varia_returns=0
    i=0
    j=0
    k=0
    z=0
    ct=0
    seg_img=[]
    variable=0
    lit=[]


    org=img
    while i<widthz:
        if (xary[i]<3 and k==0):
            k=1
            z=i
            while xary[z]<3:
                z+=1
            i=z
            if i!=0:
                j=i-1
            else:
                j=i
        elif xary[i]<3 and k==1:
            
            variable=0
            ia=i
            ja=j
            if j>=2:
                ja=j-2
            if i<=(width-1)-2:
                ia=i+2
            
            lit.append(org[0:heightss,ja:ia])
            ct+=1
            z=i
            while z<widthz and xary[z]<3:
                z+=1
            i=z
            j=i-1
            continue
        i+=1
    j=0
    i=0

    while i<len(lit):
        lit[i]=cv2.cvtColor(lit[i],cv2.COLOR_BGR2GRAY)
        i+=1

    train=[]
    i=1
    j=0
    while j<=9:
        i=1
        while i<=30:
            k=str(j)+chr(96+i)+".jpg"#(add the name of folder directory which contains training)
            name=''.join(k)
            img=cv2.imread(name,0)
            img=cv2.resize(img, (20, 20))
            a,img=cv2.threshold(img,128,255,cv2.THRESH_BINARY)
            train.append(img)
            i+=1
        j+=1

    i=1
    while i<=29:
        k='a'+str(i)+".jpg"
        name=''.join(k)
        img=cv2.imread(name,0)
        img=cv2.resize(img, (20, 20))
        a,img=cv2.threshold(img,128,255,cv2.THRESH_BINARY)
        train.append(img)
        i+=1


    i=1
    while i<=23:
        k='b'+str(i)+".jpg"
        name=''.join(k)
        img=cv2.imread(name,0)
        img=cv2.resize(img, (20, 20))
        a,img=cv2.threshold(img,128,255,cv2.THRESH_BINARY)
        train.append(img)
        i+=1

    i=1
    while i<=26:
        k='c'+str(i)+".jpg"
        name=''.join(k)
        img=cv2.imread(name,0)
        img=cv2.resize(img, (20, 20))
        a,img=cv2.threshold(img,128,255,cv2.THRESH_BINARY)
        train.append(img)
        i+=1

    i=1
    while i<=16:
        k='d'+str(i)+".jpg"
        name=''.join(k)
        img=cv2.imread(name,0)
        img=cv2.resize(img, (20, 20))
        a,img=cv2.threshold(img,128,255,cv2.THRESH_BINARY)
        train.append(img)
        i+=1

    i=1
    while i<=16:
        k='e'+str(i)+".jpg"
        name=''.join(k)
        img=cv2.imread(name,0)
        img=cv2.resize(img, (20, 20))
        a,img=cv2.threshold(img,128,255,cv2.THRESH_BINARY)
        train.append(img)
        i+=1

    i=1
    while i<=13:
        k='f'+str(i)+".jpg"
        name=''.join(k)
        img=cv2.imread(name,0)
        img=cv2.resize(img, (20, 20))
        a,img=cv2.threshold(img,128,255,cv2.THRESH_BINARY)
        train.append(img)
        i+=1

    i=1
    while i<=41:
        k='g'+str(i)+".jpg"
        name=''.join(k)
        img=cv2.imread(name,0)
        img=cv2.resize(img, (20, 20))
        a,img=cv2.threshold(img,128,255,cv2.THRESH_BINARY)
        train.append(img)
        i+=1

    i=1
    while i<=15:
        k='h'+str(i)+".jpg"
        name=''.join(k)
        img=cv2.imread(name,0)
        img=cv2.resize(img, (20, 20))
        a,img=cv2.threshold(img,128,255,cv2.THRESH_BINARY)
        train.append(img)
        i+=1

    i=1
    while i<=17:
        k='i'+str(i)+".jpg"
        name=''.join(k)
        img=cv2.imread(name,0)
        img=cv2.resize(img, (20, 20))
        a,img=cv2.threshold(img,128,255,cv2.THRESH_BINARY)
        train.append(img)
        i+=1

    i=1
    while i<=9:
        k='j'+str(i)+".jpg"
        name=''.join(k)
        img=cv2.imread(name,0)
        img=cv2.resize(img, (20, 20))
        a,img=cv2.threshold(img,128,255,cv2.THRESH_BINARY)
        train.append(img)
        i+=1

    i=1
    while i<=26:
        k='k'+str(i)+".jpg"
        name=''.join(k)
        img=cv2.imread(name,0)
        img=cv2.resize(img, (20, 20))
        a,img=cv2.threshold(img,128,255,cv2.THRESH_BINARY)
        train.append(img)
        i+=1

    i=1
    while i<=8:
        k='l'+str(i)+".jpg"
        name=''.join(k)
        img=cv2.imread(name,0)
        img=cv2.resize(img, (20, 20))
        a,img=cv2.threshold(img,128,255,cv2.THRESH_BINARY)
        train.append(img)
        i+=1

    i=1
    while i<=17:
        k='m'+str(i)+".jpg"
        name=''.join(k)
        img=cv2.imread(name,0)
        img=cv2.resize(img, (20, 20))
        a,img=cv2.threshold(img,128,255,cv2.THRESH_BINARY)
        train.append(img)
        i+=1

    i=1
    while i<=13:
        k='n'+str(i)+".jpg"
        name=''.join(k)
        img=cv2.imread(name,0)
        img=cv2.resize(img, (20, 20))
        a,img=cv2.threshold(img,128,255,cv2.THRESH_BINARY)
        train.append(img)
        i+=1

    i=1
    while i<=12:
        k='p'+str(i)+".jpg"
        name=''.join(k)
        img=cv2.imread(name,0)
        img=cv2.resize(img, (20, 20))
        a,img=cv2.threshold(img,128,255,cv2.THRESH_BINARY)
        train.append(img)
        i+=1

    i=1
    while i<=20:
        k='r'+str(i)+".jpg"
        name=''.join(k)
        img=cv2.imread(name,0)
        img=cv2.resize(img, (20, 20))
        a,img=cv2.threshold(img,128,255,cv2.THRESH_BINARY)
        train.append(img)
        i+=1

    i=1
    while i<=23:
        k='s'+str(i)+".jpg"
        name=''.join(k)
        img=cv2.imread(name,0)
        img=cv2.resize(img, (20, 20))
        a,img=cv2.threshold(img,128,255,cv2.THRESH_BINARY)
        train.append(img)
        i+=1

    i=1
    while i<=14:
        k='t'+str(i)+".jpg"
        name=''.join(k)
        img=cv2.imread(name,0)
        img=cv2.resize(img, (20, 20))
        a,img=cv2.threshold(img,128,255,cv2.THRESH_BINARY)
        train.append(img)
        i+=1

    i=1
    while i<=16:
        k='u'+str(i)+".jpg"
        name=''.join(k)
        img=cv2.imread(name,0)
        img=cv2.resize(img, (20, 20))
        a,img=cv2.threshold(img,128,255,cv2.THRESH_BINARY)
        train.append(img)
        i+=1

    i=1
    while i<=17:
        k='v'+str(i)+".jpg"
        name=''.join(k)
        img=cv2.imread(name,0)
        img=cv2.resize(img, (20, 20))
        a,img=cv2.threshold(img,128,255,cv2.THRESH_BINARY)
        train.append(img)
        i+=1

    i=1
    while i<=28:
        k='z'+str(i)+".jpg"
        name=''.join(k)
        img=cv2.imread(name,0)
        img=cv2.resize(img, (20, 20))
        a,img=cv2.threshold(img,128,255,cv2.THRESH_BINARY)
        train.append(img)
        i+=1


    train=np.array(train)
    train=train.reshape(-1,400).astype(np.float32)

    l=np.arange(10)
    label1=np.repeat(l,30)

    label2=(31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,36,36,36,36,36,36,36,36,36,36,36,36,36,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,40,40,40,40,40,40,40,40,40,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,42,42,42,42,42,42,42,42,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,44,44,44,44,44,44,44,44,44,44,44,44,44,45,45,45,45,45,45,45,45,45,45,45,45,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,48,48,48,48,48,48,48,48,48,48,48,48,48,48,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,)
    label2=np.array(label2)

    labels=np.concatenate((label1,label2),axis=0)
    labels=labels[:,np.newaxis]

    if j1111==1:
        
        knn=cv2.ml.KNearest_create()
        
        forest = RandomForestClassifier(n_estimators = 130)#
        
        svm= SVC(kernel='poly')

        nn = MLPClassifier(hidden_layer_sizes=(270,150, ),activation='logistic',max_iter=500,verbose=False)

        print("training kNN model")
        knn.train(train, cv2.ml.ROW_SAMPLE, np.ravel(labels))
        print("training Random Forest model")
        forest = forest.fit( train, np.ravel(labels))#
        print("training Neural Network model")
        nn.fit(train,np.ravel(labels))
        print("training SVM model")
        svm.fit(train,np.ravel(labels))

    outp=[]
    outp1=[]
    outp2=[]
    outp3=[]
    for test in lit:
        ht,wd=test.shape
        if ht<15:
            continue 

        a,test=cv2.threshold(test,128,255,cv2.THRESH_BINARY)
        i=0
        j=0
        count=0
        while i<ht:
            j=0
            while j<wd:
                if test[i][j]==0:
                    count+=1
                j+=1
            i+=1

        if count<95:
            continue

        test=cv2.resize(test, (20, 20))
        a,test=cv2.threshold(test,128,255,cv2.THRESH_BINARY)

        test=np.array(test)
        test=test.reshape(-1,400).astype(np.float32)
        
        ret,result,neighbours,dist=knn.findNearest(test,k=3)
        result1 = forest.predict(test)
        result2= nn.predict(test)
        result3= svm.predict(test)
        
        if (dist[0][0]>5500000 and dist[0][1]>5500000) or (dist[0][0]>5500000 and dist[0][2]>5500000) or (dist[0][1]>5500000 and dist[0][2]>5500000):
            continue
        else:
            if result[0][0]==31.0:
                outp.append('A')
            elif result[0][0]==32.0:
                outp.append('B')
            elif result[0][0]==33.0:
                outp.append('C')
            elif result[0][0]==34.0:
                outp.append('D')
            elif result[0][0]==35.0:
                outp.append('E')
            elif result[0][0]==36.0:
                outp.append('F')
            elif result[0][0]==37.0:
                outp.append('G')
            elif result[0][0]==38.0:
                outp.append('H')
            elif result[0][0]==39.0:
                outp.append('I')
            elif result[0][0]==40.0:
                outp.append('J')
            elif result[0][0]==41.0:
                outp.append('K')
            elif result[0][0]==42.0:
                outp.append('L')
            elif result[0][0]==43.0:
                outp.append('M')
            elif result[0][0]==44.0:
                outp.append('N')
            elif result[0][0]==45.0:
                outp.append('P')
            elif result[0][0]==46.0:
                outp.append('R')
            elif result[0][0]==47.0:
                outp.append('S')
            elif result[0][0]==48.0:
                outp.append('T')
            elif result[0][0]==49.0:
                outp.append('U')
            elif result[0][0]==50.0:
                outp.append('V')
            elif result[0][0]==51.0:
                outp.append('Z')
            else:
                outp.append(int(result[0][0]))

        if (dist[0][0]>5500000 and dist[0][1]>5500000) or (dist[0][0]>5500000 and dist[0][2]>5500000) or (dist[0][1]>5500000 and dist[0][2]>5500000):
            continue
        else:
            if result1[0]==31.0:
                outp1.append('A')
            elif result1[0]==32.0:
                outp1.append('B')
            elif result1[0]==33.0:
                outp1.append('C')
            elif result1[0]==34.0:
                outp1.append('D')
            elif result1[0]==35.0:
                outp1.append('E')
            elif result1[0]==36.0:
                outp1.append('F')
            elif result1[0]==37.0:
                outp1.append('G')
            elif result1[0]==38.0:
                outp1.append('H')
            elif result1[0]==39.0:
                outp1.append('I')
            elif result1[0]==40.0:
                outp1.append('J')
            elif result1[0]==41.0:
                outp1.append('K')
            elif result1[0]==42.0:
                outp1.append('L')
            elif result1[0]==43.0:
                outp1.append('M')
            elif result1[0]==44.0:
                outp1.append('N')
            elif result1[0]==45.0:
                outp1.append('P')
            elif result1[0]==46.0:
                outp1.append('R')
            elif result1[0]==47.0:
                outp1.append('S')
            elif result1[0]==48.0:
                outp1.append('T')
            elif result1[0]==49.0:
                outp1.append('U')
            elif result1[0]==50.0:
                outp1.append('V')
            elif result1[0]==51.0:
                outp1.append('Z')
            else:
                outp1.append(int(result1[0]))

        if (dist[0][0]>5500000 and dist[0][1]>5500000) or (dist[0][0]>5500000 and dist[0][2]>5500000) or (dist[0][1]>5500000 and dist[0][2]>5500000):
            continue
        else:
            if result2[0]==31.0:
                outp2.append('A')
            elif result2[0]==32.0:
                outp2.append('B')
            elif result2[0]==33.0:
                outp2.append('C')
            elif result2[0]==34.0:
                outp2.append('D')
            elif result2[0]==35.0:
                outp2.append('E')
            elif result2[0]==36.0:
                outp2.append('F')
            elif result2[0]==37.0:
                outp2.append('G')
            elif result2[0]==38.0:
                outp2.append('H')
            elif result2[0]==39.0:
                outp2.append('I')
            elif result2[0]==40.0:
                outp2.append('J')
            elif result2[0]==41.0:
                outp2.append('K')
            elif result2[0]==42.0:
                outp2.append('L')
            elif result2[0]==43.0:
                outp2.append('M')
            elif result2[0]==44.0:
                outp2.append('N')
            elif result2[0]==45.0:
                outp2.append('P')
            elif result2[0]==46.0:
                outp2.append('R')
            elif result2[0]==47.0:
                outp2.append('S')
            elif result2[0]==48.0:
                outp2.append('T')
            elif result2[0]==49.0:
                outp2.append('U')
            elif result2[0]==50.0:
                outp2.append('V')
            elif result2[0]==51.0:
                outp2.append('Z')
            else:
                outp2.append(int(result2[0]))

        if (dist[0][0]>5500000 and dist[0][1]>5500000) or (dist[0][0]>5500000 and dist[0][2]>5500000) or (dist[0][1]>5500000 and dist[0][2]>5500000):
            continue
        else:
            if result3[0]==31.0:
                outp3.append('A')
            elif result3[0]==32.0:
                outp3.append('B')
            elif result3[0]==33.0:
                outp3.append('C')
            elif result3[0]==34.0:
                outp3.append('D')
            elif result3[0]==35.0:
                outp3.append('E')
            elif result3[0]==36.0:
                outp3.append('F')
            elif result3[0]==37.0:
                outp3.append('G')
            elif result3[0]==38.0:
                outp3.append('H')
            elif result3[0]==39.0:
                outp3.append('I')
            elif result3[0]==40.0:
                outp3.append('J')
            elif result3[0]==41.0:
                outp3.append('K')
            elif result3[0]==42.0:
                outp3.append('L')
            elif result3[0]==43.0:
                outp3.append('M')
            elif result3[0]==44.0:
                outp3.append('N')
            elif result3[0]==45.0:
                outp3.append('P')
            elif result3[0]==46.0:
                outp3.append('R')
            elif result3[0]==47.0:
                outp3.append('S')
            elif result3[0]==48.0:
                outp3.append('T')
            elif result3[0]==49.0:
                outp3.append('U')
            elif result3[0]==50.0:
                outp3.append('V')
            elif result3[0]==51.0:
                outp3.append('Z')
            else:
                outp3.append(int(result3[0]))
    out= ''.join(map(str,outp))
    print("knn "+out)
    out1= ''.join(map(str,outp1))
    print("RF  "+out1)
    out2= ''.join(map(str,outp2))

    global outstring

    outstring = out1

    print("NN  "+out2)
    out3= ''.join(map(str,outp3))
    print("SVM "+out3)

    break
    j1+=1


image = Image.open(master.filename)
image = image.resize((160, 120), Image.ANTIALIAS) 
t = ImageTk.PhotoImage(image)

panel = Label(master, image = t)
panel.pack()
panel.place(x="200",y="70")


master.minsize(600,500)
master.geometry("320x100")
master.configure(background="#FFE4B5")


label1 = Label(master, text= "AUTOMATIC NUMBER PLATE RECOGNITION SYSTEM",  font='Helvetica 22 bold',borderwidth=2, relief="groove")   
label1.pack()
label1.place(x=12,y=10)

labelq = Label(master, text= "Input image path",font='Helvetica 17 ',borderwidth=5, relief="groove")
labelw = Label(master, text= "Output" , font='Helvetica 17 ',borderwidth=5, relief="groove")
labelq.pack() 
labelw.pack() 
labelq.place(x=35,y=240)
labelw.place(x=35,y=280)

label1 = Label(master, text= master.filename, borderwidth=2, font='Helvetica 17 bold',relief="groove")   
label1.pack()
label1.place(x=200,y=240)


label = Label(master, text= outstring, borderwidth=2, font='Helvetica 17 bold',relief="groove")   
label.pack() 
label.place(x=200,y=280)

master.mainloop( ) 

