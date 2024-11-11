import subprocess
import precision_recall

with open('training_measures.txt','a') as file :
    file.write("New test\n")

n_epochs = 0

subprocess.run(['python','train.py']+['--epochs=10'])
n_epochs += 10
subprocess.run(['python','generate_baseline.py'])

precision, recall = precision_recall.compute_precision_recall('samples_baseline')
   
with open('training_measures.txt','a') as file :
    file.write(f"Number of epochs : {n_epochs}\n")
    file.write(f"precision : {precision}\n")
    file.write(f"recall : {recall}\n")
        
for i in range(100):

    subprocess.run(['python','train.py']+['--train_from_checkpoint=True --epochs=10'])
    n_epochs += 10
    subprocess.run(['python','generate_baseline.py'])

    precision, recall = precision_recall.compute_precision_recall('samples_baseline')
    
    with open('training_measures.txt','a') as file :
        file.write(f"Number of epochs : {n_epochs}\n")
        file.write(f"precision : {precision}\n")
        file.write(f"recall : {recall}\n")