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