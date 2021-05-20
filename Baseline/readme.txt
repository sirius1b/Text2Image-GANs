*****************************************************************************
To test the code                                                             
*****************************************************************************
1. Open cmd/terminal in cwd                                                  
2. run the command 'pip install -r requirements.txt'                        
3. run the command 'python downloader.py' to download the dataset
    --> If this does not work then download the dataset from the link
    -->'https://drive.google.com/file/d/1EgnaTrlHGaqK5CCgHKLclZMT_AMSTyh8/view'
    --> and add the dataset in 'dataset' directory.
    --> The size of dataset is 1.87GB.
4. run the command 'python tester.py' to get results for both baseline models
-----------------------------------------------------------------------------

*****************************************************************************
Directory Structure
*****************************************************************************
1. dataset ==> has the flower dataset in hdf5 format
2. modelState ==> Stores the trained models
    --> dcgan-cls.pth ==> DCGAN-CLS model
    --> gan-cls-int.pth ==> GAN-CLS-INT model
3. results ==> stores sample results i.e. expected v/s generated images
4. src ==> the training process
-----------------------------------------------------------------------------

Note : 
** This code uses pytorch for deep learning based implementations.
** Please install pytorch according to the specification of your machine.
** The code will only run on device which has 'cuda' enabled.
** Do not alter directory structure as paths are relative for downloading 
    and extraction of dataset.
-----------------------------------------------------------------------------
Abhinav Sharma  (2018002)
Lavanya Verma   (2018155)
Pranav Sharma   (2018169)
-----------------------------------------------------------------------------