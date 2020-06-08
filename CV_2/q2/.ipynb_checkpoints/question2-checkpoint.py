{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import cv2\n",
    "\n",
    "criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)\n",
    "objp = np.zeros((12*12,3))\n",
    "objp=objp.astype(np.float32)\n",
    "\n",
    "objp[:,:2] = np.mgrid[0:12,0:12].T.reshape(-1,2)\n",
    "\n",
    "objpoints = []\n",
    "imgpoints = []\n",
    "\n",
    "for i in range(15):\n",
    "    gray = cv2.imread('Left'+str(i+1)+'.bmp',0)\n",
    "    ret, corners = cv2.findChessboardCorners(gray, (12,12),None)\n",
    "    if ret != False:\n",
    "        objpoints.append(objp)\n",
    "        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)\n",
    "        imgpoints.append(corners2)\n",
    "        img = cv2.drawChessboardCorners(gray, (12,12), corners2,ret)\n",
    "        cv2.imwrite('img'+str(i+2),gray\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 0.99573573  0.01418558  0.09115442]\n",
      " [-0.017317    0.9992835   0.03365429]\n",
      " [-0.0906117  -0.0350893   0.99526793]]\n",
      "[[ 0.99461295  0.00328862  0.10360628]\n",
      " [-0.00705469  0.99932673  0.03600431]\n",
      " [-0.10341813 -0.03654127  0.99396651]]\n",
      "[[ 0.49478634 -0.82391258  0.27632326]\n",
      " [ 0.76197362  0.56420911  0.31790608]\n",
      " [-0.41783092  0.05325545  0.90696261]]\n",
      "[[ 0.22614681 -0.96482522  0.1340519 ]\n",
      " [ 0.87903868  0.14284301 -0.45484819]\n",
      " [ 0.41970063  0.22069927  0.88042218]]\n",
      "[[-0.01895633 -0.79359318  0.60815337]\n",
      " [ 0.9984009   0.01737502  0.05379352]\n",
      " [-0.05325685  0.6082006   0.79199479]]\n",
      "[[-0.15877775 -0.7234425   0.6718784 ]\n",
      " [ 0.96605096  0.02663453  0.256975  ]\n",
      " [-0.2038018   0.68987068  0.69465335]]\n",
      "[[ 0.68375321  0.00535385 -0.72969369]\n",
      " [-0.03683294  0.99895162 -0.02718456]\n",
      " [ 0.72878315  0.04546429  0.68323357]]\n",
      "[[ 0.94876042  0.03516012  0.31403414]\n",
      " [-0.04064804  0.99911361  0.01094246]\n",
      " [-0.31337104 -0.02314665  0.94934863]]\n",
      "[[ 0.95901599  0.03591875  0.28106613]\n",
      " [-0.05854383  0.99564748  0.07251694]\n",
      " [-0.27723807 -0.08599959  0.95694468]]\n",
      "[[ 0.8800717  -0.13991192  0.45376035]\n",
      " [ 0.01223873  0.96197137  0.27287598]\n",
      " [-0.47468307 -0.23459697  0.84831612]]\n",
      "[[ 0.93435813  0.04893952 -0.35295864]\n",
      " [ 0.08273006  0.93366486  0.34846185]\n",
      " [ 0.34659864 -0.35478845  0.86832859]]\n"
     ]
    }
   ],
   "source": [
    "parameters = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)\n",
    "for i in parameters[3]:\n",
    "    result=cv2.Rodrigues(i)\n",
    "    print(result[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.03846702661187315\n",
      "0.027302261323137404\n",
      "0.0630259471307111\n",
      "0.030858197469598933\n",
      "0.03226222469211951\n",
      "0.33431143018724696\n",
      "0.2912240883308437\n",
      "0.10980369652759304\n",
      "0.3036072968911244\n",
      "0.050053152497678555\n",
      "0.3712769920644868\n",
      "Mean error:  0.15019930124785577\n"
     ]
    }
   ],
   "source": [
    "mean_error = 0\n",
    "for i in range(len(objpoints)):\n",
    "    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)\n",
    "    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)\n",
    "    print (error)\n",
    "    mean_error =mean_error+ error\n",
    "\n",
    "print (\"Mean error: \", mean_error/len(objpoints))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
