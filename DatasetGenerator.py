## Import packages

from time import time
from PIL import Image
import os
import requests
from shutil import copyfile, copyfileobj
import tarfile

## Load Config  : 

# INDOOR CONFIG 

### keywords with the suffix _DIR are parsed in such a way that new folders are created (if they do not already exists) 
### Please keep this scheme also for SUN CONFIG 

config_dict = dict(ROOT_DIR = '/content',
                    PROJECT_DIR = '/content/drive/MyDrive/AML project 2021-2-F1801Q151',
                    LOCAL_DIR = '/content/indoor/',
                    REMOTE_DIR = '/content/drive/My Drive/AML project 2021-2-F1801Q151/datasets/',
                    DATASET_URL = 'http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar',
                    TRAINING_URL = 'https://web.mit.edu/torralba/www/TrainImages.txt' ,
                    TEST_URL = 'https://web.mit.edu/torralba/www/TestImages.txt',
                    DATASET_LOCAL = '/content/indoor/indoorCVPR_09.tar',
                    DATASET_REMOTE = '/content/drive/MyDrive/AML project 2021-2-F1801Q151/datasets/indoorCVPR_09.tar',
                    FILE_NAME = 'indoorCVPR_09.tar',
                    TRAIN_IMAGES_TXT = '/content/drive/MyDrive/AML project 2021-2-F1801Q151/datasets/TrainImages.txt',
                    TEST_IMAGES_TXT = '/content/drive/MyDrive/AML project 2021-2-F1801Q151/datasets/TestImages.txt'
                    
)

## Classes and functions


class Configurator(object):
  """
  Imports configuration data as attributes. If config file contains new directories pathnames, they will be created.

  :param initial_data: object containing config data
  :type initial_data: dict 
  
  """

  def __init__(self, initial_data: dict):
        for key in initial_data:
            setattr(self, key, initial_data[key])
            self._createFolder(initial_data[key], key = key )
  
  
  def _createFolder(self, name: str, key: str) :
    if key.endswith('_DIR'):
      if not os.path.exists(name):
        print(f'{name} folder has been created')
        os.mkdir(name)


class DataLoader(Configurator):
  """
  Loads data with transfer() and extract() methods. Its attributes are inherited by Configurator class according to a config_dict

  :param CONFIGURATION: configuration object that contains configuration data (i.e. pathnames, urls, etc...) 
  :type CONFIGURATION: dict 
   
  """

  def __init__(self,CONFIGURATION):
    super().__init__(CONFIGURATION)
  
  def get_from_gDrive(self):
    """
    Transfers a .tar file content from Google Drive to a local folder (specified in config_dict) 
    """
    t1 = time()
    if not os.path.isdir(self.LOCAL_DIR + 'Images'):
      copyfile(self.DATASET_REMOTE, self.DATASET_LOCAL)
      t2 = time()
      self._print_timing(t1,t2,operation='transfer')
    else:
      print('File already transfered!')
  
  def get_unfold(self):
    """
    Unzip a .tar file and extract its content 
    """
    t1 =  time()
    os.chdir(self.LOCAL_DIR)
    tar = tarfile.open(self.DATASET_REMOTE)
    tar.extractall()
    tar.close()
    #os.remove(self.DATASET_REMOTE)
    os.chdir(self.ROOT_DIR)
    t2 = time()
    self._print_timing(t1,t2,operation='extraction')

  def backup(self):
    """
    Backups a dataset copying and saving it to Google Drive 
    """
    os.chdir(self.ROOT_DIR)
    tar = tarfile.open(self.FILE_NAME, 'w')
    tar.add(self.FILENAME.split('.')[0])
    tar.close()
    copyfile(self.DATASET_LOCAL, self.DATASET_REMOTE)
    #os.remove(self.DATASET_REMOTE)


  def download(self, url: str, isTxtFile = False) : 
    """
    Downloads data from URL. 

    :param url: data source url 
    :type url: str 
    :isTxtFile: specify if the file to download is in .txt format . Default False 
    :type isTxtFile: bool
    """
    filename = url.split('/')[-1]   #get the last part of URL string
    if not os.path.isfile(self.LOCAL_DIR + '/' + filename):
      os.chdir(self.LOCAL_DIR)
    if isTxtFile : 
      with requests.get(url) as r:
        with open(filename, 'wb') as f:
          f.write(r.content)
    else : 
      with requests.get(url, stream=True) as r:
        with open(filename, 'wb') as f:
          copyfileobj(r.raw, f)
      
      os.chdir(self.ROOT_DIR)
    
    return filename


  def get_images_pathnames_list(self, file_pathname):
    """
    Parse a file and get a list containing images pathnames
    
    :param file_pathname: pathname
    :type file_pathname: str

    :return: collection of images pathnames 
    :rtype: list
    
    """

    self.image_paths = list()
    with open(file_pathname, mode='r') as paths:
      lines = paths.readlines()
      for line in lines:
        self.image_paths.append(line.replace('\n', ''))
    
    return self.image_paths

  def _print_timing(self,start,end,operation):
    print(f'File {operation} completed in {round((end-start),3)} seconds')
    
class Generator(DataLoader) : 
  def __init__(self, CONFIGURATION,
               new_image_size = (224,224)) : 
    
    super().__init__(CONFIGURATION)
    self.new_image_size = new_image_size
    
  def generate(self,resample=Image.BICUBIC, training_set_flag = True ):
    """
    Generates dataset by rotating and transforming images 

    :param resample:
    :type resample:
    :param training_set_flag: select which dataset images refer to. Default is True, that means images names list refers to training set, conversely test set is generated.
    :type training_set_flag: bool
    """

    start = time()
    if training_set_flag : 
      file_txt = self.TRAIN_IMAGES_TXT
    else : 
      file_txt = self.TEST_IMAGES_TXT

    images_pathnames_list = self.get_images_pathnames_list(file_txt)
    for image_path in images_pathnames_list:
      original_image = self._open_image(self.LOCAL_DIR + 'Images/' + image_path)   #  opening image
      image_name = image_path.split('/')[1]                            
      resulting_image = self._resize(original_image,resample)                        # resizing image
      for index, angle in enumerate([0, 90, 180, 270]):
        dest_folder = self._create_rotation_folder(angle, training_set_flag )  # creating destination folder
        resulting_image = self._rotate(resulting_image, rotation_degree = angle)                 # rotating image by four directions
        resulting_image = self._to_RGB(resulting_image)                                 # converting image to RGB, if it is not 
        self._save_image(resulting_image ,image_name, dest_path = dest_folder)           # saving image to its directory based on rotation degree
    end = time()
    self._print_timing(start,end, operation= 'generation process' ) 
    print('Generation process completed !')

  
  def _rotate(self,image, rotation_degree: int):
    """
    
    :param rotation_degree: select the rotation angle among 90,180 and 270 
    :type rotation_degree: int

    Returns : 
      [type] : [description]
    
    """
    if rotation_degree == 0 : 
      return image 
    else : 
      rotations = {
                      90  : Image.ROTATE_90, 
                      180 : Image.ROTATE_180,
                      270 : Image.ROTATE_270
                  }
      image = image.transpose(method=rotations.get(rotation_degree))
      return image

  def _resize(self, image, resample):
    """
    :param resample:
    :type resample: 

    Returns 
      [type] : [descr]
    """
    image = image.resize(self.new_image_size, 
                         resample)
    return image

  def _to_RGB(self,image):
    """
    Converts non-RGB images, if existing,  into RGB

    :param image:
    :type image:   

    Returns :
        PIL.Image.Image : image 
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

  def _open_image(self,image_path: str):
    """
    Opens an image with PIL package

    :param image_path:
    :type image_path:

    Returns : 
      PIL.JpegImagePlugin.JpegImageFile : image 
    """
    image = Image.open(image_path, mode='r')
    return image

  def _save_image(self,image: str, image_name,  dest_path: str):
    image.save(dest_path+image_name)

  def _create_rotation_folder(self, rotation_angle: int, training_set_flag: bool):
    if training_set_flag: 
      data_set_type = 'TRAINING_SET/'
    else : 
      data_set_type = 'TEST_SET/'
    os.chdir(self.LOCAL_DIR)
    if not os.path.exists(data_set_type+str(rotation_angle)):
      if not os.path.exists(f"{self.LOCAL_DIR+data_set_type}"):
        os.mkdir(f"{self.LOCAL_DIR+data_set_type}")
      os.mkdir(f"{self.LOCAL_DIR+data_set_type+str(rotation_angle)}")
      print(f"{self.LOCAL_DIR+data_set_type+str(rotation_angle)} folder has been created")
    
    return f"{self.LOCAL_DIR+data_set_type+str(rotation_angle)}/"

## Load data

data = DataLoader(config_dict)
data.get_from_gDrive()
data.get_unfold()

## Generate dataset

generator = Generator(config_dict)

generator.generate(training_set_flag = True)
generator.generate(training_set_flag = False)
