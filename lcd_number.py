# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import cv2


class Digits:

    def __init__(self,imagefile):
        #
        #IMAGES
        self.originalimg=0
        self.workingimg=0
        #FRAMED IMAGES
        self.framecoords=[]
        self.framedoriginalimg=0
        self.framedworingimg=0
        #loadimage
        if isinstance(imagefile,type(np.array([]))):
            self.im_newimagearray(imagefile)
        else:
            self.im_newimage(imagefile)
        #ProcessedImagesDIGITS
        self.digits=self.im_getdigitsr()
        self.processdigits=self.im_getprocesseddigits()
        self.im_eraselowcountd()


    ######LoadImage case file, case array
    def im_newimage(self,imagefile):
        self.originalimg=cv2.imread(imagefile,0)
        self.workingimg=self.originalimg
    def im_newimagearray(self,imagearray):
        gray= cv2.cvtColor(imagearray, cv2.COLOR_BGR2GRAY)
        self.originalimg=gray
        self.workingimg=self.originalimg
    #####BAISCPROCESSINGDEFS------INPUT---npARRAY-----OUTPUT---Processednparray
    def im_dilate(self,f):
        kerneld=np.ones((16,16),np.uint8)
        dilatedimage=cv2.dilate(f,kerneld,iterations = 1)
        return dilatedimage
    def im_erode(self,f):
        kernele=np.ones((9,9),np.uint8)
        erodedimage=cv2.erode(f,kernele,iterations = 1)
        return erodedimage
    def im_treshold(self,f):
        ret,im=cv2.threshold(f,120,255,cv2.THRESH_BINARY)
        return im
    #BASICFEATURES-----Input---- np array
    #contours-----Input---- np array Output----list contours
    def im_contours(self,f):
        a,b=cv2.findContours(f.copy(), 1, 2)
        return a
    #boundingrectangles----Input----nparray----output----list rectangles
    def im_rectangles(self,f):
        c=self.im_contours(f)
        rectangles=[]
        for cnt in c:
            rectangles.append(cv2.boundingRect(cnt))
        return rectangles
    #GETDISPLAYFRAME----INPUT-----none---output----rectangle of display
    def im_getdisplay(self):
        f=self.workingimg
        f=self.im_treshold(f)
        f=self.im_dilate(f)
        f=self.im_erode(f)

        ccc=0
        ind2=[0,0]
        a=self.im_contours(f)
        for count,c in enumerate(a):
            cc=[]
            for m in range(len(c)-1):
                ss=c[m+1]-c[m]
                cc.append(max(ss[0]))
                if max(cc)>ccc:
                    ccc=max(cc)
                    ind2[1]=ind2[0]
                    ind2[0]=count
        rectangledisplay=cv2.boundingRect(a[ind2[1]])
        return rectangledisplay
    #GETDIGITS ----INPUT-----none---output----listdigits
    def im_getdigitsr(self):
        rectangles=[]
        digits=[]
        rectanglesmax=[]
        display=self.im_getdisplay()
        x,y,w,h=display
        foriginal=self.originalimg[y:y+h,x:x+w]
        f=self.workingimg
        f=self.im_treshold(f)
        f=self.im_dilate(f)
        f=self.im_erode(f)
        f=f[y:y+h,x:x+w]
        a,b = cv2.findContours(f.copy(), 1, 2)
        for cnt in a:
            rectangles.append(cv2.boundingRect(cnt))
        for rectangle in rectangles:
            x1,y1,w1,h1=rectangle
            if w1*h1>500:
                rectanglesmax.append(rectangle)

        rectanglesmax=sorted(rectanglesmax, key=lambda x: x[0])



        for rectangle in rectanglesmax:
            x1,y1,w1,h1=rectangle
            digits.append(foriginal[y1:y1+h1,x1:x1+w1])
        return digits
    def im_getdigits(self):
        return self.digits
    #apply treshold dilate digits
    def im_getprocesseddigits(self):
        pdigits=[]
        for m in self.digits:
            c=self.im_treshold(m)
            pdigits.append(c)
        return pdigits


    #COUNTPOINTSDIGITS INPUT---listarrays output----counts per item
    def im_count(self):
        count=[]
        for i in self.processdigits:
            c=0
            for m in i:
                c=c+sum(m)/255
            count.append(c)
        return count
    def im_eraselowcountd(self):
        s1=[]
        s2=[]
        count=self.im_count()

        for num,i in enumerate(self.processdigits):
            if count[num]>60:
                s1.append(i)
        for num,i in enumerate(self.digits):
            if count[num]>60:
                s2.append(i)
        self.processdigits=s1
        self.digits=s2
        #LINECOUNTING
    def im_hitcount(self):
        digits=self.processdigits
        hitxf=[]
        hityf=[]
        hittt=[]
        for count,i in enumerate(self.processdigits):
            digito1=digits[count]
            h=len(digito1[:,0])
            w=len(digito1[0,:])
            coordsy=np.array(range(1,3))*h/(len(np.array(range(1,3)))+1)
            coordsx=np.array(range(1,2))*w/(len(np.array(range(1,2)))+1)

            m=0
            cc=0
            hits=[]
            for i in range(h):
                trig=digito1[i,coordsx[0]]
                if trig==0:
                    cc=0
                if trig!=0 and cc==0:
                    hits.append([i,coordsx[0]])
                    m=m+1
                    cc=1
            hityf.append(hits)


            mx=0
            ccx=0
            hitstx=[]
            hitsx=[]
            mxt=[]
            for i in range(2):
                for s in range(w):
                    trig=digito1[coordsy[i],s]
                    if trig==0:
                        ccx=0
                    if trig!=0 and ccx==0:
                        hitsx.append([coordsy[i],s])
                        mx=mx+1
                        ccx=1
                hitstx.append(hitsx)
                mxt.append(mx)
                mx=0
                hitsx=[]
            hitxf.append(hitstx)
            hittt.append([hits,hitstx])

        return hittt
    #zoneClasification


    def im_zoneclassification(self):
        digitaldicts=[]
        digitaldict={'b1':0,'b2':0,'b3':0,'b4':0,'b5':0,'b6':0,'b7':0,}
        for i in range(len(self.processdigits)):
            digitaldicts.append(dict(digitaldict))
        digits=self.processdigits
        hits=self.im_hitcount()
        for count,i in enumerate(digits):
            hity=hits[count][0]
            hitx=hits[count][1]
            limx=len(i[0,:])/2
            limxy=len(i[:,0])/2
            limy1,limy2=[len(i[:,0])/3,2*len(i[:,0])/3]

            for hit in hity:
                if hit[0]<limy1:
                    digitaldicts[count]['b2']=1
                elif hit[0]>=limy1 and hit[0]<=limy2:
                    digitaldicts[count]['b7']=1
                elif hit[0]>limy2:
                    digitaldicts[count]['b5']=1
            for hit in hitx:
                for q in hit:
                    if q[1]<=limx and q[0]<limxy:
                        digitaldicts[count]['b1']=1
                    elif q[1]<=limx and q[0]>=limxy:
                        digitaldicts[count]['b6']=1
                    if q[1]>limx and q[0]<limxy:
                        digitaldicts[count]['b3']=1
                    elif q[1]>limx and q[0]>=limxy:
                        digitaldicts[count]['b4']=1

        return digitaldicts
    def im_intreturn(self):
        fulldigitdict={0:{'b1':1,'b2':1,'b3':1,'b4':1,'b5':1,'b6':1,'b7':0},1:{'b1':0,'b2':1,'b3':0,'b4':0,'b5':1,'b6':0,'b7':0},2:{'b1':0,'b2':1,'b3':1,'b4':0,'b5':1,'b6':1,'b7':1},3:{'b1':0,'b2':1,'b3':1,'b4':1,'b5':1,'b6':0,'b7':1},4:{'b1':1,'b2':0,'b3':1,'b4':1,'b5':0,'b6':0,'b7':1},5:{'b1':1,'b2':1,'b3':0,'b4':1,'b5':1,'b6':0,'b7':1},6:{'b1':1,'b2':0,'b3':0,'b4':1,'b5':1,'b6':1,'b7':1},7:{'b1':0,'b2':1,'b3':1,'b4':1,'b5':0,'b6':0,'b7':0},8:{'b1':1,'b2':1,'b3':1,'b4':1,'b5':1,'b6':1,'b7':1},9:{'b1':1,'b2':1,'b3':1,'b4':1,'b5':0,'b6':0,'b7':1}}
        s=self.im_zoneclassification()
        m=[]
        for i in s:
            match=0
            for count in range(len(fulldigitdict)):
                q=fulldigitdict[count]
                if q==i:
                    m.append(count)
                    match=1
            if match==0:
                m.append(1)
        #join digits
        m=map(str,m)
        m=''.join(m)
        if m:
            m=int(m)
        return m

        #PLOTING
    def im_plotrawarray(self,digits):
        m=digits
        plt.imshow(m,cmap=plt.cm.gray)
    def im_plotdigits(self):
        digits=self.im_getdigits()
        fig, axes = plt.subplots(nrows=1, ncols=len(digits),figsize=(16,6))
        for count,ax in enumerate(axes):
            ax.imshow(digits[count],cmap=plt.cm.gray)
    def im_plotdigitsbinary(self):
        digits=[]
        digits2=self.processdigits
        for i in digits2:
            digits.append(self.im_treshold(i))
        fig, axes = plt.subplots(nrows=1, ncols=len(digits),figsize=(16,6))
        for count,ax in enumerate(axes):
            ax.imshow(digits[count],cmap=plt.cm.gray)
    def plot(self):
        plt.show()




#EXAMPLE 
direccion='test_images/26.png'

num=Digits(direccion)

digits=num.processdigits
num.im_plotdigitsbinary()



#num.im_plotdigitsbinary()
########CALCULOHITS

num.plot()
print num.im_intreturn()


#num.plot()


#num.im_plotdigitsbinary()
#num.plot()
    #########
