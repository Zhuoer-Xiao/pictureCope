import cv2
import numpy as np
import math
import sewar as sw

def compare_psnr(mse):
    return 10*np.log10(255**2/mse)

def color_cast(image_path=None):
    # image_path = r'D:\test\1.jpg'
    frame = cv2.imread(image_path)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(img)
    h, w, _ = img.shape
    da = a_channel.sum()/(h*w)-128
    db = b_channel.sum()/(h*w)-128
    hist_a = [0]*256
    hist_b = [0]*256
    for i in range(h):
        for j in range(w):
            ta = a_channel[i][j]
            tb = b_channel[i][j]
            hist_a[ta] += 1
            hist_b[tb] += 1
    msq_a = 0
    msq_b = 0
    for y in range(256):
        msq_a += float(abs(y-128-da))*hist_a[y]/(w*h)
        msq_b += float(abs(y - 128 - db)) * hist_b[y] / (w * h)

    result = math.sqrt(da*da+db*db)/math.sqrt(msq_a*msq_a+msq_b*msq_b)
    print("d/m = %s" % result)






def lvbo(src):
    imgInfo=src.shape
    height=imgInfo[0]
    width=imgInfo[1]
    dst=np.zeros((height,width,3),np.uint8)
    for i in range(3,height-3):
        for j in range(3,width-3):
            sumb=0
            sumg=0
            sumr=0
            for m in range(-3,3):
                for n in range(-3,3):
                    (b,g,r)=img[i+m,j+n]
                    sumb=sumb+int(b)
                    sumg=sumg+int(g)
                    sumr=sumr+int(r)
                b=np.uint8(sumb/36)
                g=np.uint8(sumg/36)
                r=np.uint8(sumr/36)
                dst[i,j]=(b,g,r)
    return dst

def zmMinFilterGray(src, r=7):
    return cv2.erode(src, np.ones((2 * r + 1, 2 * r + 1)))  # 使用opencv的erode函数更高效


def guidedfilter(I, p, r, eps):
    height, width = I.shape
    m_I = cv2.boxFilter(I, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p

    m_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I

    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a * I + m_b


def getV1(m, r, eps, w, maxV1):  # 输入rgb图像，值范围[0,1]
    '''计算大气遮罩图像V1和光照值A, V1 = 1-t/A'''
    V1 = np.min(m, 2)  # 得到暗通道图像
    V1 = guidedfilter(V1, zmMinFilterGray(V1, 7), r, eps)  # 使用引导滤波优化
    bins = 2000
    ht = np.histogram(V1, bins)  # 计算大气光照A
    d = np.cumsum(ht[0]) / float(V1.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()
    V1 = np.minimum(V1 * w, maxV1)  # 对值范围进行限制

    return V1, A


def deHaze(m, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):
    Y = np.zeros(m.shape)
    V1, A = getV1(m, r, eps, w, maxV1)  # 得到遮罩图像和大气光照
    for k in range(3):
        Y[:, :, k] = (m[:, :, k] - V1) / (1 - V1 / A)  # 颜色校正
    Y = np.clip(Y, 0, 1)
    if bGamma:
        Y = Y ** (np.log(0.5) / np.log(Y.mean()))  # gamma校正,默认不进行该操作
    return Y

def update(image,lightness,saturation):
	image = cv2.imread('defog.jpg', cv2.IMREAD_COLOR).astype(np.float32) / 255.0
	hlsImg = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
	# 1.调整亮度（线性变换)
	hlsImg[:, :, 1] = (1.0 + lightness / float(100)) * hlsImg[:, :, 1]
	hlsImg[:, :, 1][hlsImg[:, :, 1] > 1] = 1
	# 饱和度
	hlsImg[:, :, 2] = (1.0 + saturation / float(100)) * hlsImg[:, :, 2]
	hlsImg[:, :, 2][hlsImg[:, :, 2] > 1] = 1
	# HLS2BGR
	lsImg = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR) * 255
	lsImg = lsImg.astype(np.uint8)
	return lsImg

def makeupred(img):
    imginfo=img.shape
    height=imginfo[0]
    width=imginfo[1]
    b, g, r = cv2.split(img)
    for i in range(1, int(height)):
        for j in range(1, width):
            r[i][j]+=30
    dst=cv2.merge([b,g,r])
    return dst

def makeupred1(img):
    (b, g, r) = cv2.split(img)
    rH = cv2.equalizeHist(r)
    gH = cv2.equalizeHist(g)
    result = cv2.merge((b, g, rH))  # 通道合成
    return result

def ImageHist(image,type):
    color=(255,255,255)
    windowName='Gray'
    if type==31:
        color=(255,0,0)
        windowName='B Hist'
    elif type==32:
        color=(0,255,0)
        windowName='G Hist'
    elif type==33:
        color=(0,0,255)
        windowName='R Hist'
    hist=cv2.calcHist([image],[0],None,[256],[0.0,255.0])#1 img 2 通道 3 mask蒙版 4 size 5 像素值
    minV,maxV,minL,maxL=cv2.minMaxLoc(hist)
    histImg=np.zeros([256,256,3],np.uint8)
    for h in range(256):
        intenNormal=int(hist[h]*256/maxV)
        cv2.line(histImg,(h,256),(h,256-intenNormal),color)
    cv2.imshow(windowName,histImg)
    return histImg


def trans(img):
    imginfo=img.shape
    b,g,r=cv2.split(img)
    height=imginfo[0]
    width=imginfo[1]
    r_total=0
    g_total=0
    for i in range(0,height):
        for j in range(0,width):
            r_total+=(r[i][j])
    for i in range(0,height):
        for j in range(0,width):
            g_total+=(r[i][j])
    for i in range(0,height):
        for j in range(0,width):
            r[i][j]=(g_total/r_total*r[i][j])
    img=cv2.merge([b,g,r])
    return img

def hist(img,a):
    b,g,r=cv2.split(img)
    if  (a=='b'):
        b=cv2.equalizeHist(b)
    if (a == 'g'):
        g = cv2.equalizeHist(g)
    if (a == 'r'):
        r = cv2.equalizeHist(r)
    return cv2.merge([b,g,r])

def trans3(img):#nxn模板
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    dst = np.zeros((height, width, 3), np.uint8)
    for i in range(4, height - 4):
        for j in range(4, width - 4):
            sumg = 0
            sumr = 0
            for m in range(-4, 4):
                for n in range(-4, 4):
                    (b, g, r) = img[i + m, j + n]
                    sumr = sumr + r
                    sumg = sumg + g
                    r = np.uint8(sumg / sumr)
                dst[i, j] = (b, g, r)

    return dst

def trans4(src,dst):
    imginfo = src.shape
    height = imginfo[0]
    width = imginfo[1]
    sb,sg,sr=cv2.split(src)
    db,dg,dr=cv2.split(dst)
    g1=0
    r1=0
    for i in range(0,height):
        for j in range(0,width):
            g1 += dg[i][j] - sg[i][j]
            r1 += dr[i][j] - sr[i][j]
    dr=cv2.multiply(dr,g1/r1)
    dst=cv2.merge([db,dg,dr])
    return dst

def showhist(img):
    (b,g,r)=cv2.split(img)
    channels=cv2.split(img)#获取通道RGB->R  G  B
    for i in range(0,3):
        ImageHist(channels[i],31+i)
    cv2.waitKey(0)

if __name__ == '__main__':
    src=cv2.imread('6.jpg')   #原图src

    m = deHaze(src / 255.0) * 255
    cv2.imwrite('defog.jpg', m)
    img=cv2.imread('defog.jpg',1) #处理后的图img
    cv2.imshow('result1',img)
    cv2.imshow('src',src)
    color_cast('defog.jpg')
    dst1=trans4(src, img)
    cv2.imwrite('dst1.jpg', dst1)
    color_cast('dst1.jpg')
    cv2.imshow('dst',dst1)
    showhist(img)
    showhist(trans4(src,img))
    cv2.waitKey(0)






