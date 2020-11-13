If you just want to use the AI and see what it generates then first open and modify 'GAN Training.py' for where you saved the dataset. then run 'GAN Training.py' and after it has trained and saved the AI model run ganTest.py it should work. if you want to create and train your own AI based on
My code then go in to GAN Training.py and take a look at it then do whatever modifications you need.

This uses Python 3.79 64bit requires Tesnorflow 2, heras, matplotlib and numpy.

if you get a memory error like "exceeds 10% system memory" you either have a dataset that is too big or not enough ram om your computer.

it is recommended to get gpu support for tensorflow as it runs much MUCH faster.

feel free to try different datasets and model architectures for the generator and discriminator. 

how does this AI work?
this AI is actually 2 different AIs working against each other the generator and the discriminator. the discriminator is being trained
to tell the difference betweeen the 'real' images from the dataset and the 'fake images' from the generator. while the generator tries
to fool the discriminator. so the generator model never sees the actual dataset only the discriminator does.

if you need more infomation or help the internent is full of people much more knoledgable than me.