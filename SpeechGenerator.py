"""

A generator for reading and serving audio files

https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html

Remember to use multiprocessing:

# Train model on dataset

model.fit_generator(generator=training_generator,

                    validation_data=validation_generator,

                    use_multiprocessing=True,

                    workers=6)

                    

"""



import numpy as np

import keras

from extractMFCC import computeFeatures1



class SpeechGen(keras.utils.Sequence):

    """

    'Generates data for Keras'

    

    list_IDs - list of files that this generator should load

    labels - dictionary of corresponding (integer) category to each file in list_IDs

    

    Expects list_IDs and labels to be of the same length

    """

    def __init__(self, list_IDs, labels, root_dir, batch_size = 64, dim = 16000, shuffle = True):

        'Initialization'

        self.dim = dim

        self.batch_size = batch_size

        self.labels = labels

        self.root_dir = root_dir

        self.list_IDs = list_IDs

        self.shuffle = shuffle

        self.on_epoch_end()



    def __len__(self):

        'Denotes the number of batches per epoch'

        return int(np.floor(len(self.list_IDs) / self.batch_size))



    def __getitem__(self, index):

        'Generate one batch of data'

        # Generate indexes of the batch

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]



        # Find list of IDs

        list_IDs_temp = [self.list_IDs[k] for k in indexes]



        # Generate data

        X, y = self.__data_generation(list_IDs_temp, indexes)



        return X, y



    def on_epoch_end(self):

        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)



    def __data_generation(self, list_IDs_temp, indexes):

        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Initialization

        X1 = np.empty((self.batch_size, self.dim))

        X = np.empty((self.batch_size, 99, 39))

        y = np.empty((self.batch_size), dtype=int)

        # 

        # Generate data

        for i in range(len(list_IDs_temp)):

            """

            #load data from file, saved as numpy array on disk

            print(i)

            print(list_IDs_temp[i])

            print(self.labels[indexes[i]])

            if(list_IDs_temp[i].endswith('np')):

                print("case 0")

                list_IDs_temp[i] = list_IDs_temp[i] + 'y'

            elif(list_IDs_temp[i].endswith('n')):

                print("case 1")

                list_IDs_temp[i] = list_IDs_temp[i] + 'py'

            elif(list_IDs_temp[i].endswith('.')):

                print("case 2")

                list_IDs_temp[i] = list_IDs_temp[i] + 'npy'

            elif(list_IDs_temp[i].endswith('v')):

                print("case 3")

                list_IDs_temp[i] = list_IDs_temp[i] + '.npy'

            """   

            #curX = np.load(list_IDs_temp[i], allow_pickle=True)

            curX = np.load(self.root_dir + "/" + list_IDs_temp[i])

            #normalize

            #invMax = 1/(np.max(np.abs(curX))+1e-3)

            #curX *= invMax            

            

            #curX could be bigger or smaller than self.dim

            if curX.shape[0] == self.dim:

                X1[i] = curX

                #print('Same dim')

            elif curX.shape[0] > self.dim: #bigger

                #we can choose any position in curX-self.dim

                randPos = np.random.randint(curX.shape[0]-self.dim)

                X1[i] = curX[randPos:randPos+self.dim]

                #print('File dim bigger')

            else: #smaller

                randPos = np.random.randint(self.dim-curX.shape[0])

                X1[i,randPos:randPos+curX.shape[0]] = curX

                #print('File dim smaller')

            

            features = computeFeatures1(X1[i], Fc = 16000)

            features = features/np.max(features)

            

            # Store class

            X[i] = features

            y[i] = self.labels[indexes[i]]

        

        X = X.reshape((self.batch_size, 99, 39, 1))

        return X, y