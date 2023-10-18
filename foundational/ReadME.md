I decided to start this over "from scratch", at least inasmuch as I needed to document
the process much better than I have been. To that end, I'll create a new "clean" version
that I will outline here. Unless otherwise noted, all files are in hd5 format. The
"key" for those formats will be noted in the respective file(s) that create them.

The following pre-work is documented here, but the code isn't in the best shape to show currently.

10 - Discover the global min/max
	 Drop any unnecessary columns (pressure for example)
	 Set column type(s) to be most efficient further on in the code
20 - Use that min/max to normalize the velocity data
30 - Create files that are split by time (so 1200 per experiment). 
	 These are named "time".hdf in the respective directory (of that experiment)

OK, so how do we get (for any given coordinate x,y,z) within one of the files that is already
seperated by time that we will look to encode?

Within the "AnalyzerOfCoordinates.py" there is an "inner class" called:
class TestAnalyzerOfCoordinates(unittest.TestCase):
This shows you the basics for 
        x, y, z = -78.0, 23.0, 3.0
In 3p6 for an arbitrary time of "300", but this is more of a "are all the values there"
test, not "what is the order" of a list.

This portion:
   def test_provide_coordinates_ordered_list(self):
Shows how with a given centroid of
        x, y, z = -113.0, -68.0, -17.0

The full example of which is shown as a JSON object:
/Users/kkreth/PycharmProjects/cgan/dataset/3p6 time 300 sample at ReadME.txt.json

Anyway, you can see when looking at the element order, it can be confirmed using the assert
in this test, that the list is ordered as expected.

In this manner, given a time (arbitrary here, but it is how the files are broken up), given an
experiment ID, and given an x,y,z, you can use this class to find the 125 coordinates (assuming 
they exist).

Probably worth taking a minute to look at the training files used for the AE. Those files have an
"attribute" I believe it is called in HD5 files, this is a unique key. I'll be writing a class to 
read these files, I'll call that:

read_ae_training_files_fileloader.py

I'll pick (probably the very first one in the file) of the combined attribute file (which took a day
or so to gather on the Unity cluster to sample from all experiments and all times). The file is roughly
(excepting the fact that duplicates are dropped) ~1000 samples (thanks to random duplicates this is
closer to ~985) for each time slice of each experiment. Since there are >1,000 time slices this amounts
to more than a million samples from each and every experiment, "evenly" distributed across time.
If an encoder can be built using this data that can both have a very low reconstruction error (which has
to be on the order of hundredths I would think based on the precision of the input data) as measured
through MSE, and KL divergence (which is proven to be useful to follow on so-called downstream 
algorithms, whether they are CNNs, Transformers, or Random Forests).

It is worth noting here, that if this AE fails to perform as required in the downstream side, we will
have to revisit the criterion conceit above. Specifically, measuring MSE across all 125 datapoints may be
detrimental (given our MSE precision) using 125 datapoints (think ~a hundred datapoints and hundredths of precision,
these might overlap with one another). We might need to adjust the loss function to instead minimize the one centroid
error, either completely or using a forced importance through something like a log transformation assumption in the
underlying polynomial. Hopefully it won't come to this, as I certainly don't want to regenerate all of this data
iteratively, but honestly whether this works or not this is something we should explore for completeness. Folks will
definitely care if it works, and if it doesn't this might be the "central reason".



40 - Create "sub-files" that only contain the actual target for training the AE
	 These are only slightly smaller in terms of coordinates, as you have to move in
	 2 spaces from the edge (where you can no longer get the 125 coordinates). This is
	 where we will create the [3,125] velocity sets.
	 This go around the files will have the list of coordinates mapped to each set
	 of data, making "going back" that much easier
	 These files are named "time"_exhaustive_tensors.hd5
50 - Substantial sample from all files compiled here to train the AE
	 This fill will be called:
	 AE_Training_Input.hd5
60 - Train the AE
	 Model files will be saved as:
	 ptv_convolutional_encoder_"EPOCH".pth
70 - Run the AE against all data (and check that it is between -1 and 1)
	 Files here will be called:
	 1000_tensor_for_transformer.torch.gz
	 Note: These are proper TORCH files, not hd5