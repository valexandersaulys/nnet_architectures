# Chapter 1: Why Keras?

Neural networks are increasingly becoming part of a connected world. From the
self-learning agents playing go from Google Deepmind to the insanely accurate
systems utilized by Facebook to match friends, neural networks are becoming a
hot commodity in today's environment. They represent a powerful new way to
build computer systems that were not previously thought possible. That is, they
can help create computer systems that self-learn.

Neural networks are a subclass of machine learning. Machine learning agents are
capable of self-learning. By self-learning, we mean an agent that is capable of
looking at previous examples to some problem and being able to look at new
examples and "solve" them. Solve can mean a large variety of things, which will
be gotten into later, but for now let's use an example. Suppose we wish to create
an agent that categorizes email as spam or not spam. One way to
do this is to design an algorithm that searches for key words or phrases and uses
those to classify new incoming emails. However, this is can be very difficult;
after all, what constitutes spam email or not? Its hard for a human to answer this
question.

Machine learning recommends a different approach. If we were to design an agent
using machine learning principals, we could gather up a series of data that is
marked spam or not spam. This data would also contain the text of the email in
question. From this, we could use a series of algorithms, namely machine learning
algorithms, to "learn" the rules that it finds are most useful for determining
whether or not email is spam.

This powerful shift in thinking is what is propeling a series of new innovations
across multiple industries. For instance, computer vision was a notoriosuly hard
problem to solve but in recent years has seen huge strides thanks to machine
learning. It was difficult to solve because few knew what to look for when trying
to identify a unique image in a way a computer does it. Humans just instinctively
know how to do this and nobody is quite sure why. By using machine learning
algorithms, one can pass in a series of previous examples and have the computer
learn what's important.

In particular, neural networks are a powerful subclass of machine learning
algorithm. They utilize long sequences of weights and functions to solve some
sort of problem. What makes them more powerful than other types of machine
learning algorithms, and ultimately why they are so popular, is that they do not
(generally) require prior knowledge. By prior knowledge, we can mean a lot of
different ideas, but for the sake of this chapter we'll define prior knowledge
as outside, human inflected information. An example of this can be seen in the
spam problem. Instead of passing in the raw text of the email, we might try to
pass in some statistic about the email instead. This could be how often the words
"viagara" appears. These stats are called _handcrafted features_ and they're
generally not considered to be good. This is because we take out own human biases
into a problem, which may or may not help for the most optimal outcome.

In contrast, neural networks do not need this prior knowledge. One can pass in the
raw text and just let the neural network find the most optimal "features" it
considers important. We could also pass in the handcrafted features if we wanted.
It should be noted that neural networks are not exclusive in this ability, but
what further makes them useful is that they are flexible. Neural networks have
been successfully used in many many applications already. Some examples are
listed below:

* TO-DO
* fdsa
* fads

These are not the only examples either. A simple online search will reveal quite
a staggering number of uses for neural networks.

However, there is one large caveat to neural networks. They are mighty difficult
to explain. Nobody really knows why they work so well. Geoff Hinton, a legendary
academic researcher in the field, once remarked in a reddit AMA that he had no
idea why pooling (a technique in neural networks, to be talked about later)
worked so well. This is coming from a man who has dedicated his life's work to
studying neural networks too.

This opaqueness can make neural networks daunting, but it is a bit hyperbolic to
say we know nothing. This book was written to help shine a light on suggestions
for what constitutes a good neural network and basic ideas on what architectures
work best in certain instances. Having been a top 1% Kaggler and build neural
networks for emotional intelligence companies, I can tell you that when I was
starting out two years ago, it was frustrating how little the information seemed
to talk about how to use neural networks. Too often, I'd browse through mountains
of information about the academic and theoretical work of neural networks, and not
enough on the refined, good practice tips information.

This book is by __no__ means exhaustive I should add. This is why it's being crowd
sourced as a GitBook. By pulling on the exhaustive support within the deep learning
and machine learning community, a great book on architectures can be written that
helps cover a wide spread of people interested in this sort of work. If you're reading
this and would like to contribute in any way, whether that be by correcting
spelling errors to writing whole new sections or correcting where I might be
wrong, please feel free to clone this repo (on either Github, Gitlab, or Gitbook)
and make the necesary changes. All I ask is that you push them back so that
other people may use them.

As a side note, I often use neural networks and deep learning interchangeably.
Though they are not necesarily the same thing, it is fair to consider them
for the sake of varied language in this book. 

TO-DO: Insert what license I'm going to use.



### Purpose of a Neural Network Library

There are many, many neural network libraries these days. A few years back,
there were practically none. This development is good. Many readers here will
no doubt know why libraries are good. It allows one to piggy back common
functions off of some other project. This saves time and allows for specialization
on those topics. Neural networks are no exception here.

Prior to using neural network specific libraries, researchers often used linear
algebra libraries for help and assistance when coding. They were able to simplify
the complex matrix work that deep learning often requires. But they lacked a few
major features.

For one, they relied on the CPU. For two, they were often not comprehensive enough
as deep learning progressed. For three, they only worked on one machine. These
problems were all solved with further improvements. 

Deep learning is a resource intensive process. Especially with moore's law
starting to give out, its become important to eek out more power from a machine.
GPUs are often leveraged to do this as a result. However, GPUs are difficult to
work with and are not really optimized to do work outside of graphics. Some
libraries do exist to use OpenCL and OpenGL, but NVIDIA has thankfully cut
around this tough work by implementing something called CUDA. CUDA is a parallel
platform for computing that uses NVIDIA graphics cards to do mass computations.
Modern CPUs will have up to eight cores with threading, but GPUs like the
Geforce GTX 980 can have up to 3072 cores. This type of processing power requires
a whole new way of thinking about computers, but for the purposes of this book
we can gloss over this and simply use the CUDA libraries. This CUDA library
allows for tremendous gains in speeds. For example, batch training a digita
recognizing machine will take 100-200 seconds on an Intel Skylake CPU while only
10-20 seconds on a Gfore GTX. This allows one to save time, crucial when training
models.

However, CUDA is not all that comprehensive. It really provides a shallow layer
on which other libraries and programs are expected to further build. Therefore,
more comprehensive networks are needed. One example of this is Theano. Developed
out of the MILA labs at the University de Montreal in Montreal, Quebec, Canada,
this library allowed for powerful gains in research. No longer did one have to
create a convolution in 2d space algorithm for implementing a convolutional
neural network, they could simply call a function from Theano to do this. 



### Enter Keras

Keras is best thought of as a framework from which to hang neural network
architectures. ....

Also explain tensorflow as opposed to theano


### How to Install

###### Linux


###### Mac OS X


###### Windows


###### Scikit-learn and other libraries