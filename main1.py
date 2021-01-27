import os
import shutil
import random
import numpy as np
import math
from keras.engine.input_layer import Input
from keras import backend as K
import pandas as pd
import root_pandas

from models1 import set_trainability, make_generator, make_discriminator, make_gan
from plotting1 import plot_losses, plot_distribution, plot_correlation

###Recreate plots folders
folders_=['plots1', 'plotcorr1']
for dir in folders_:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)


###Only on
input_columns = ['trk_pt', 'trk_eta', 'trk_phi']
output_columns = ['trk_dxyClosestPV', 'trk_dzClosestPV', 'trk_ptErr', 'trk_etaErr',
                  'trk_dxyErr', 'trk_dzErr', 'trk_nChi2']
temp = ['trk_isTrue', 'trk_algo']
input_file = "data/trackingNtuple_TTBarLeptons.root"
#dataframe = root_pandas.read_root(input_file,columns=input_columns+output_columns+temp, flatten=True)[input_columns+output_columns+temp]
dataframe = root_pandas.read_root(input_file,columns=input_columns+output_columns+temp, flatten=True, chunksize=10000).__iter__().next()[input_columns+output_columns+temp]
dataframe = dataframe[(dataframe.trk_isTrue==1) & (dataframe.trk_algo==4)]
dataframe = dataframe.drop(temp, axis=1)

"""
generate_input:
    
This generates random samples and stacks them into columns 'pt', 'eta' and 'phi' and the stack is returned
"""

def generate_input(n_samples):
    pT = np.random.uniform(0.0, 100.0, n_samples)
    eta = np.random.uniform(-2.1, 2.1, n_samples)
    phi = np.random.uniform(-math.pi, math.pi, n_samples)

    result = np.column_stack((pT, eta, phi))
    return result

"""

sample_data_and_gen:

'sample_data_and_gen' generates some input data and the 'generated' data is the prediction from G.
Remember that 'generated' contains essentially 3 columns of the variables 'pt', 'eta' and 'phi'.

Then it concatenates the input and generated onto the same thing, but two separate vectors (axis = 1).

Variable 'real' is real data that is read from the root file.
Variable 'output' is a dataframe stacked from the generated and real data. Look at the columns
Variable 'truth' is an array consiting of 2 vectors: [1, 1, ..., 1] [-1, -1, ..., -1]

This function returns two dataframes with the following columns:

Columns1 : 'trk_pt', 'trk_eta', 'trk_phi', 'trk_dxyColsesPV', 'trk_dzClosestPV', 'trk_ptErr', 'trk_etaErr', 
            'trk_dxyErr', 'trk_dzErr', 'trk_nChi2'
Columns2: 'trk_generated'

"""

def sample_data_and_gen(G, n_samples):
    inp = generate_input(n_samples/2)
    generated = G.predict(inp)
    generated = np.concatenate((inp, generated), axis=1)
    real = dataframe.sample(n=n_samples/2) # Real data 
    output = pd.DataFrame(np.concatenate((generated, real), axis=0), columns=input_columns+output_columns)
    truth = np.concatenate((np.ones(n_samples/2),-1*np.ones(n_samples/2)))
    output['trk_generated'] = truth

    # print real
    # print len(inp[:,0])
    
    output = output.sample(frac=1.0) # What does this do?

    return output[input_columns+output_columns], output['trk_generated']

"""
pretain:
    
X is a dataframe that consists of the columbs 'trk_pt', 'trk_eta' etc as in Columns1
y is also a dataframe that contains the colum 'generated samples' as in Columns2
"""

def pretrain(G, D, n_samples, batch_size=32, epochs=10):
    X, y = sample_data_and_gen(G, n_samples)
    set_trainability(D, True)
    D.fit(X[output_columns], y, epochs=epochs, batch_size=batch_size, verbose=False)
    
"""

train:
    
First iterate over e_range which is by default 100 and call 'pretain' each time

    In each of these iterations we do another loop of size 20000/64 = 312 normally
    We set the variables 'X' and 'y' to be dataframes as they are called with 'sample_data_and_gen'
    We append to d_loss a thing D.train_on_batch. What this function does is just a single gradient update
    Afterwards generate some samples and again train_on_batch. 

"""

#def train(GAN, G, D, epochs=100, n_samples=20000, batch_size=64, verbose=False, v_freq=10):
def train(GAN, G, D, epochs=100, n_samples=20000, batch_size=64, verbose=False, v_freq=1):
    d_iters=10
    D_loss = []
    G_loss = []
    e_range = range(epochs)
    for epoch in e_range:
        d_loss = []
        g_loss = []
        pretrain(G, D, n_samples, batch_size, 20)
        for batch in range(n_samples/batch_size):
            X, y = sample_data_and_gen(G, batch_size)
            set_trainability(D, True)
            """            
            print "\ntrk_pt:\n"
            
            print X.trk_pt
            
            print "\ntrk_pt generated:\n"
            
            print X[y==1]['trk_pt']
            
            print "\ntrk_pt real:\n"
            
            print X[y==-1]['trk_pt']
            
            print "\ntrk_eta generated:\n"
            
            print X[y==1]['trk_eta']
            
            print "\n\trk_eta real:n"
            
            print X[y==-1]['trk_eta']
                
            print "\ny:\n"
            
            print y
            """
            
            d_loss.append(D.train_on_batch(X[output_columns], y))

            set_trainability(D, False)
            X = generate_input(batch_size)
            y = -1*np.ones(batch_size) #Claim these are true tracks, see if discriminator believes
            g_loss.append(GAN.train_on_batch(X, y))

        G_loss.append(np.mean(g_loss))
        D_loss.append(np.mean(d_loss))
        if verbose and (epoch + 1) % v_freq == 0:
            print("Epoch #{}: Generative Loss: {}, Discriminative Loss: {}".format(epoch + 1, G_loss[-1], D_loss[-1]))
            plot_losses(G_loss, D_loss)  
            X, y = sample_data_and_gen(G, 2000)
            binning = np.linspace(-0.5,0.5,100)
            for distr in output_columns:
                plot_distribution(X[y==1][distr], binning, epoch=epoch+1, title="Generated_"+distr)
                
              #  pt_valuesG = X[y==1]['trk_pt'].values
             #   pt_valuesR = X[y==-1]['trk_pt'].values
                
                if (epoch + 1) % 50 == 0:
                
                    # Plot each pt-pt, pt-eta, phi-phi etc. correlation
                    for distr1 in input_columns + output_columns:
                        
                        valuesGen1 = X[y==1][distr1].values
                        valuesReal1 = X[y==-1][distr1].values
                        
                        for distr2 in input_columns + output_columns:
                            
                            valuesGen2 = X[y==1][distr2].values
                            valuesReal2 = X[y==-1][distr2].values
                            
                            
                            plot_correlation(valuesGen1, valuesGen2, epoch = epoch + 1, title='Generated ' + distr1 + ' and ' + distr2)
    
                            if epoch + 1 == 50:
                                plot_correlation(valuesReal1, valuesReal2, epoch = epoch + 1, title='Real ' + distr1 + ' and ' + distr2)
                    
                    
                if epoch == 1:
                    plot_distribution(X[y==-1][distr], binning, epoch = epoch+1, title="Real "+distr)
                                        

        if (epoch + 1) % 200 == 0:
            print "Old lr: "+ str(K.eval(D.optimizer.lr))
            K.set_value(D.optimizer.lr, 0.5*K.eval(D.optimizer.lr))
            K.set_value(G.optimizer.lr, 0.5*K.eval(G.optimizer.lr))
            print "New lr: "+ str(K.eval(D.optimizer.lr))

#            D.optimizer.lr.set_value(0.1*D.optimizer.lr.get_value())
#            G.optimizer.lr.set_value(0.1*G.optimizer.lr.get_value())

    return D_loss, G_loss

if __name__ == '__main__':
    G_in = Input(shape=[len(input_columns)])
    G, G_out = make_generator(G_in, len(output_columns), lr=1e-4)
    D_in = Input(shape=[len(output_columns)])
    D, D_out = make_discriminator(D_in, lr=1e-4)

    GAN_in = Input(shape=[len(input_columns)])
    GAN, GAN_out = make_gan(GAN_in, G, D)
    
    print "Start pretraining"
    pretrain(G, D, 10000)
    print "Start training"
    train(GAN, G, D, 1000, verbose=True)

