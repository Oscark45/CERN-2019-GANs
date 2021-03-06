import matplotlib
matplotlib.use('Agg')
import seaborn
import matplotlib.pyplot as plt
import numpy as np

def plot_losses(G_loss, D_loss):
    epochs = range(len(G_loss))
    plt.plot(epochs, G_loss, label='Generative loss')
    plt.plot(epochs, D_loss, label='Discriminative loss')
    plt.plot(epochs, np.add(D_loss,G_loss), label='Total loss')
#    plt.yscale('log')
    plt.title('Losses at epoch '+str(len(G_loss)))
    plt.legend()
    plt.xlabel('Epochs')
    plt.savefig('plots1/Losses.pdf')
    plt.clf()

def plot_distribution(values, binning, epoch, title="", xlabel=""):
    plt.hist(values, bins=binning)
    plt.title(title+" at epoch "+str(epoch))
    plt.xlabel(xlabel)
    plt.savefig('plots1/'+title+'_epoch_'+str(epoch)+'.pdf')
    plt.clf()
    
def plot_correlation(vector1, vector2, epoch, title = ""):
        
    corr = np.corrcoef(vector1, vector2)
    
    plt.scatter(vector1, vector2)
    plt.title(title + " epoch " + str(epoch) + '.pdf')
    plt.show()
    plt.savefig('plotcorr1/' + title + '_epoch_' + str(epoch) + '.pdf')
    plt.clf()
    
    print corr
    
    pass


