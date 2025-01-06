'''
OK, so this should show you how to gather results to compare to the original.
We will use some data from the following (relative path):
/Users/kkreth/PycharmProjects/cgan/dataset/test_latent.csv

So, the first entry there is:

[-117, 79, -25]	5	[0.4195218086242676, -0.7442097067832947, -0.03907422348856926, 0.34993523359298706, -0.3007224202156067, -0.08908732235431671, 0.9461111426353455, -0.8101040124893188, -0.7684844136238098, -0.7359620928764343, 0.861623227596283, 0.2510451078414917, -0.22258543968200684, 1.0600299835205078, 0.060479436069726944, -0.2816435694694519, -0.1564859300851822, 0.6778295636177063, 0.29230621457099915, 0.8608457446098328, -0.6570100784301758, 0.8620972633361816, -0.43942326307296753, 0.19025230407714844, -0.3213137984275818, -0.7652362585067749, -0.7213377356529236, -0.9439583420753479, 0.7047829031944275, 0.11300221085548401, 0.21458157896995544, -0.6653024554252625, 0.29231420159339905, 0.5367957353591919, -0.5104660391807556, -0.6565309762954712, 0.9986112713813782, -0.31548523902893066, -0.7990534901618958, -0.8184934854507446, 0.6317899227142334, 0.9586987495422363, 0.6700903177261353, 0.8809971213340759, -0.632462203502655, -0.3602072596549988, 0.4141937792301178, -0.4831354022026062]

The "5" above is for "time", which from a "decoder" standpoint is meta-data or otherwise superfluous.

We need to break this down into two parts:

Part One
--------
Get the coordinate data...here all we need to know is the experiment, in this case 3p6 and the "centroid"...which
in our case will be the "middle" value of the 125 returned...but more on that later.

So, the information we will need is:
"3p6" and "[-117, 79, -25]"
which we can interpret as:

x = -117
y = 79
z = -25

We can feed this coordinate (and experiment name) to the class:

And get back the 125 coordinate results that will map to our decode of the 48 float latent result, in this case
stored on each row for the:
/Users/kkreth/PycharmProjects/cgan/dataset/test_latent.csv
file.

'''