
# coding: utf-8

# # Problem 1

# Nothing really to answer here. Just note use of `load_mat` and `test_train_split` which do everything for you! When shuffling I set `random_state=0` for reproducible results.
# 

# In[1]:

import matplotlib.pyplot as plt
import numpy as np

import scipy,numpy,sklearn,matplotlib
from scipy import io

from sklearn import model_selection,svm
from sklearn.model_selection import train_test_split,KFold

matplotlib.rcParams['figure.figsize'] = (18,16)


# In[2]:

cifar = scipy.io.loadmat("hw01_data/hw01_data/cifar/train")
#print(cifar)
cifar_train, cifar_v = train_test_split(cifar['trainX'],test_size = 5000, random_state = 0)
#print(cifar_train.shape)
#print(cifar_v.shape)
#print(cifar['trainX'].shape)
cifar_train_data = cifar_train[:,0:-1]
cifar_train_target = cifar_train[:,-1]
cifar_v_data = cifar_v[:,0:-1]
cifar_v_target = cifar_v[:,-1]


# In[3]:

spam = scipy.io.loadmat("hw01_data/hw01_data/spam/spam_data")
#print(spam)
#print(spam['training_data'].shape,np.ravel(spam['training_labels']).shape)

spam_train_data,spam_v_data,spam_train_target,spam_v_target = train_test_split(spam['training_data'],np.ravel(spam['training_labels']),test_size = 0.2, random_state =0)


# In[4]:

mnist = scipy.io.loadmat("hw01_data/hw01_data/mnist/train")
#print(mnist.keys())

mnist_train, mnist_v = train_test_split(mnist['trainX'],test_size = 10000, random_state = 0)
mnist_train_data = mnist_train[:,0:-1]
mnist_train_target = mnist_train[:,-1]
mnist_v_data = mnist_v[:,0:-1]
mnist_v_target = mnist_v[:,-1]


# # Problem 2

# The function `f` does all the hard work. The rest of the code is just setting up the inputs and plotting the graphs.

# In[5]:

def f(k,tdata,ttarget,vdata,vtarget,name):
    #print(k,tdata.shape)
    if name=='cifar':
        cl= svm.SVC(kernel='linear',random_state=0)
    else:
        cl = svm.LinearSVC(random_state=0)
    tdatak = tdata[0:k,:]
    ttargetk=ttarget[0:k]
    cl.fit(tdatak,ttargetk)
    return cl.score(tdatak,ttargetk),cl.score(vdata,vtarget)


# In[6]:

ks = [100,200,500,1000,2000,5000,10000]
ks2 = [100,200,500,1000,2000,4137]
ks3 = [100,200,500,1000,2000,5000]


# In[7]:

thing = [(ks,mnist_train_data,mnist_train_target,mnist_v_data,mnist_v_target,'mnist'),(ks2,spam_train_data,spam_train_target,spam_v_data,spam_v_target,'spam'),(ks3,cifar_train_data,cifar_train_target,cifar_v_data,cifar_v_target,'cifar')]

colors = iter(["red","orange","yellow","green","blue","purple"])


# In[8]:

def g(ks,tdata,ttarget,vdata,vtarget,name,colors):
    results = [f(k,tdata,ttarget,vdata,vtarget,name) for k in ks]
    tacc = [result[0] for result in results]
    vacc = [result[1] for result in results]
    #print(tacc)
    #print(vacc)
    plt.plot(ks,tacc, color=next(colors), marker ='o', linestyle='-',alpha =0.5, label =(name+ ' training'))
    plt.plot(ks,vacc, color=next(colors), marker ='o', linestyle='-',alpha =0.5, label =(name+ ' validation'))
    #print(name)
    return


# In[9]:

for (ks,tdata,ttarget,vdata,vtarget,name) in thing: 
    g(ks,tdata,ttarget,vdata,vtarget,name,colors)

plt.legend(loc="lower right")
plt.axis([50,15000,0,1.25])
plt.xscale('log')
plt.ylabel("Classification Accuracy")
plt.xlabel("Number of Training Examples")
plt.title("Problem 2")
plt.show()


# # Problem 3

# As before, the function `h` does all the work. the rest is just setting things up.

# In[10]:

def h(k,tdata,ttarget,vdata,vtarget,C):
    #print(k,tdata.shape)
    cl= svm.LinearSVC(C=C)
    tdatak = tdata[0:k,:]
    ttargetk=ttarget[0:k]
    cl.fit(tdatak,ttargetk)
    return cl.score(tdatak,ttargetk),cl.score(vdata,vtarget)


# In[11]:

tdata=mnist_train_data
ttarget=mnist_train_target
vdata=mnist_v_data
vtarget =mnist_v_target

cs = np.logspace(-10,-1,19)

cresults = [h(10000,tdata,ttarget,vdata,vtarget,c) for c in cs]
ctacc = [result[0] for result in cresults]
cvacc = [result[1] for result in cresults]

#plt.plot(cs,ctacc,color="blue")
plt.plot(cs,cvacc,color="red",marker="o")
plt.xscale('log')
plt.show()
ccc =[(cs[i],cvacc[i]) for i in range(len(cs))]
print(ccc)


# # Problem 3, C Values

# The C values are listed as c-value,validation accuraccy pairs. The best C-Value found was `3.1622776601683792e-07`.

# # Problem 4

# In[12]:

kf = KFold(n_splits = 5,shuffle=True)
spamkdata = spam['training_data']
spamklabel = np.ravel(spam['training_labels'])
#print(spamkdata.shape)
def j(c,data,label):
    score = 0
    for train_ind,test_ind in kf.split(spamkdata):
        spamktdata = data[train_ind]
        spamktlabel = label[train_ind]
        spamkvdata = data[test_ind]
        spamkvlabel = label[test_ind]
        cl = svm.LinearSVC(random_state=0,C=c)
        cl.fit(spamktdata,spamktlabel)
        score += cl.score(spamkvdata,spamkvlabel)
        #print(train_ind,test_ind)
        #print(train_ind.shape)
        #print(score)
    return score/5
a = [(c,j(c,spamkdata,spamklabel)) for c in np.logspace(-20,5,52)]
b= [x[0] for x in a]
c = [x[1] for x in a]
plt.plot(b,c,marker="o")
plt.xscale('log')
plt.show()
print(a)


# # Problem 4, C values

# The C values are listed above as c-value,validation accuraccy pairs. The best C-Value found was `11.979298107105205`.

# # Problem 5

# In[13]:

def l(k,tdata,ttarget,vdata,vtarget,C):
    #print(k,tdata.shape)
    cl= svm.LinearSVC(C=C)
    tdatak = tdata[0:k,:]
    ttargetk=ttarget[0:k]
    cl.fit(tdatak,ttargetk)
    return cl.score(vdata,vtarget)


# In[14]:

tdata=mnist_train_data
ttarget=mnist_train_target
vdata=mnist_v_data
vtarget =mnist_v_target

cs = np.logspace(-10,-1,20)

cresults = [l(500,tdata,ttarget,vdata,vtarget,c) for c in cs]

#plt.plot(cs,ctacc,color="blue")
plt.plot(cs,cresults,color="red",marker="o")
plt.xscale('log')
plt.show()


# In[15]:

def output_mnist(c,test):
    cl= svm.LinearSVC(C=c)
    mnist_trainf = mnist['trainX']
    mnist_trainf_data = mnist_trainf[:,0:-1]
    mnist_trainf_target = mnist_trainf[:,-1]
    cl.fit(mnist_trainf_data,mnist_trainf_target)
    out = cl.predict(test)
    numpy.savetxt("hw01_data/hw01_data/mnist/kaggleout.csv", numpy.column_stack((numpy.array(list(range(10000))),out)).astype(int), fmt ="%i",delimiter = ",", header = "Id,Category",comments = "")


# In[16]:

mnisttest = scipy.io.loadmat("hw01_data/hw01_data/mnist/test")


# In[17]:

output_mnist(0.000000207,mnisttest['testX'])


# # MNIST kaggle scores

# ![minst](mnistkaggle.png)

# In[18]:

spammod = scipy.io.loadmat("hw01_data/hw01_data/spam/spam_data_mod")
spammoddata = spammod['training_data']
spammodlabel = np.ravel(spammod['training_labels'])


# In[19]:

kf = KFold(n_splits = 5,shuffle=True)
#print(spamkdata.shape)
def k(c,data,label):
    score = 0
    for train_ind,test_ind in kf.split(spamkdata):
        spamktdata = data[train_ind]
        spamktlabel = label[train_ind]
        spamkvdata = data[test_ind]
        spamkvlabel = label[test_ind]
        cl = svm.LinearSVC(random_state=0,C=c)
        cl.fit(spamktdata,spamktlabel)
        score += cl.score(spamkvdata,spamkvlabel)
        #print(train_ind,test_ind)
        #print(train_ind.shape)
        #print(score)
    return score/5
a = [(c,k(c,spammoddata,spammodlabel)) for c in np.logspace(-20,5,52)]
b= [x[0] for x in a]
c = [x[1] for x in a]
plt.plot(b,c,marker="o")
plt.xscale('log')
plt.show()
print(a)


# In[20]:

def output_spam(c,test):
    cl= svm.LinearSVC(C=c)
    cl.fit(spammoddata,spammodlabel)
    out = cl.predict(test)
    numpy.savetxt("hw01_data/hw01_data/spam/kaggleout.csv", numpy.column_stack((numpy.array(list(range(5857))),out)).astype(int), fmt ="%i",delimiter = ",", header = "Id,Category",comments = "")


# In[21]:

spamftest = spammod['test_data']


# In[22]:

output_spam(100,spamftest)


# # SPAM kaggle scores

# ![spam](spamkaggle.png)

# In[ ]:



