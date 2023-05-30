import torch
import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mycolorpy import colorlist as mcp

global colors
colors = ['orchid', 'turquoise', 'tomato', 'yellowgreen']
colors = mcp.gen_color(cmap="cool",n=5)

def class_specific_kld(concept_class, nr_experiences, t0, which_class=0, perc=0,dir=None):
    kl_loss = []
    t0_classes = concept_class.seen_classes[t0].to(torch.long)
    c = t0_classes[which_class].item()

    for t in range(t0, nr_experiences):
        kld = 0
        kld += concept_class.kl_distance(t0, t, c)[0]
        kl_loss.append(kld)

    # PERFORM THE ZERO TEST
    q_uncertain = 0.5*torch.ones(size=(112,))
    zero_kld = concept_class.kl_distance(t0,t0, c, q_uncertain)[0]

    print('Zero KLD', zero_kld)
    print('KLD', kl_loss)
    plt.plot(range(t0, nr_experiences), kl_loss, marker='o', label='classes %i  at t=%i'%(c, t0), color=colors[t0] )
    plt.plot(range(t0, nr_experiences), zero_kld*np.ones(nr_experiences-t0),  label='Noise', color='black', linestyle='--')
    plt.legend()

    plt.show()
    path_fig = os.path.join(dir, 'class_%i_kld-%i.png'%(c, t0))
    plt.savefig(path_fig)
    plt.close()


def singular_kld(concept_class, nr_experiences, t0, perc=0,dir=None):

    kl_loss = []
    t0_classes = concept_class.seen_classes[t0].to(torch.long)


    for t in range(t0, nr_experiences):
        kld = 0
        for c in t0_classes:

            kld += concept_class.kl_distance(t0, t, c)[0]
        kl_loss.append(kld/len(t0_classes))

    # PERFORM THE ZERO TEST
    zero_kld = 0
    q_uncertain = 0.5*torch.ones(size=(112,))
    for c in t0_classes:
        zero_kld += concept_class.kl_distance(t0,t0, c, q_uncertain)[0]/len(t0_classes)

    print('Zero KLD', zero_kld)
    print('KLD', kl_loss)
    plt.plot(range(t0, nr_experiences), kl_loss, marker='o', label='classes %i  at t=%i'%(c, t0), color=colors[t0] )
    plt.plot(range(t0, nr_experiences), zero_kld*np.ones(nr_experiences-t0),  label='Noise', color='black', linestyle='--')
    plt.legend()

    plt.show()
    path_fig = os.path.join(dir, 'kld-%i.png'%t0)
    plt.savefig(path_fig)
    plt.close()

def real_singular_kld(concept_class, nr_experiences, t0, perc=0,dir=None):

    kl_loss = []
    t0_classes = concept_class.seen_classes[t0].to(torch.long)


    for t in range(t0, nr_experiences):
        kld = 0
        for c in t0_classes:

            kld += concept_class.kl_distance_real(t0, t, c)[0]
        kl_loss.append(kld/len(t0_classes))

    # PERFORM THE ZERO TEST
    zero_kld = 0
    q_uncertain = 0.5*torch.ones(size=(112,))
    for c in t0_classes:
        zero_kld += concept_class.kl_distance(t0,t0, c, q_uncertain)[0]/len(t0_classes)

    print('Zero KLD', zero_kld)
    print('KLD', kl_loss)
    plt.plot(range(t0, nr_experiences), kl_loss, marker='o', label='classes %i  at t=%i'%(c, t0), color=colors[t0] )
    plt.plot(range(t0, nr_experiences), zero_kld*np.ones(nr_experiences-t0),  label='Noise', color='black', linestyle='--')
    plt.legend()

    plt.show()
    path_fig = os.path.join(dir, 'kld_real-%i.png'%t0)
    plt.savefig(path_fig)
    plt.close()


def overall_kld(concept_class, nr_experiences, which_class=None, perc=0,dir=None):

    kl_concepts_at_all_times = []
    chosen_classes = []
    for t0 in range(nr_experiences):
        kl_loss = []
        kl_concepts = []
        t0_classes = concept_class.seen_classes[t0].to(torch.long)

        if which_class is not None:
            t0_classes = [t0_classes[which_class].item()]
            c = t0_classes[0]
            print('Chosen class', c)
            chosen_classes.append(c)

        for t in range(t0, nr_experiences):
            kld = 0
            kld_all = np.zeros(112)
            for c in t0_classes:
                c_kld, c_concepts = concept_class.kl_distance(t0, t, c)
                kld += c_kld
                kld_all += c_concepts/len(t0_classes)

            kl_loss.append(kld/len(t0_classes))
            kl_concepts.append(kld_all)

        kl_concepts_at_all_times.append(kl_concepts)
        # PERFORM THE ZERO TEST
        zero_kld = 0
        zero_kld_all = np.zeros(112)
        q_uncertain = 0.5*torch.ones(size=(112,))
        for c in t0_classes:
            zkld, zkld_all = concept_class.kl_distance(t0,t0, c, q_uncertain)
            zero_kld += zkld / len(t0_classes)
            zero_kld_all += zkld_all / len(t0_classes)
        print('Zero KLD', zero_kld)
        print('KLD', kl_loss)
        if which_class is None:
            plt.plot(range(t0, nr_experiences), kl_loss, marker='o', label='classes at t=%i'%t0, color=colors[t0])
            save_dir = os.path.join(dir, 'kld_data.txt')
            with open(save_dir, 'a') as file:
                for kld_data in kl_loss:
                    file.write(str(kld_data))
                    file.write('\n')

        else:
            plt.plot(range(t0, nr_experiences), kl_loss, marker='o', label='class %i at t=%i'%(c, t0), color=colors[t0])
        plt.legend()

    #plt.plot(range(0, nr_experiences), zero_kld_all[0]*np.ones(4),  label='Noise', color='black', linestyle='--')
    plt.xlabel('Experience')
    plt.ylabel('Normalized KLD')
    plt.legend()
    plt.show()



    if which_class is None:
        path_fig = os.path.join(dir, 'overall_kld.png')
        plt.savefig(path_fig)
        plt.close()
        
        return kl_concepts_at_all_times

    else:
        path_fig = os.path.join(dir, 'overall_kld_for_choice_%i.png'%which_class)
        plt.savefig(path_fig)
        plt.close()
        return kl_concepts_at_all_times, chosen_classes

def overall_kld_real(concept_class, nr_experiences, which_class=None, perc=0,dir=None):

    kl_concepts_at_all_times = []
    chosen_classes = []
    for t0 in range(nr_experiences):
        kl_loss = []
        kl_concepts = []
        t0_classes = concept_class.seen_classes[t0].to(torch.long)

        if which_class is not None:
            t0_classes = [t0_classes[which_class].item()]
            c = t0_classes[0]
            print('Chosen class', c)
            chosen_classes.append(c)

        for t in range(t0, nr_experiences):
            kld = 0
            kld_all = np.zeros(112)
            for c in t0_classes:
                c_kld, c_concepts = concept_class.kl_distance_real(t0, t, c)
                kld += c_kld
                kld_all += c_concepts/len(t0_classes)

            kl_loss.append(kld/len(t0_classes))
            kl_concepts.append(kld_all)

        kl_concepts_at_all_times.append(kl_concepts)
        # PERFORM THE ZERO TEST
        zero_kld = 0
        zero_kld_all = np.zeros(112)
        q_uncertain = 0.5*torch.ones(size=(112,))
        for c in t0_classes:
            zkld, zkld_all = concept_class.kl_distance(t0,t0, c, q_uncertain)
            zero_kld += zkld / len(t0_classes)
            zero_kld_all += zkld_all / len(t0_classes)
        print('Zero KLD', zero_kld)
        print('KLD', kl_loss)
        if which_class is None:
            plt.plot(range(t0, nr_experiences), kl_loss, marker='o', label='classes at t=%i'%t0, color=colors[t0])
        else:
            plt.plot(range(t0, nr_experiences), kl_loss, marker='o', label='class %i at t=%i'%(c, t0), color=colors[t0])
        plt.legend()

    #plt.plot(range(0, nr_experiences), zero_kld_all[0]*np.ones(4),  label='Noise', color='black', linestyle='--')
    plt.xlabel('Experience')
    plt.ylabel('Normalized KLD')
    plt.legend()
    plt.show()
    if which_class is None:
        path_fig = os.path.join(dir, 'overall_kld_real.png')
        plt.savefig(path_fig)
        plt.close()
        return kl_concepts_at_all_times

    else:
        path_fig = os.path.join(dir, 'overall_kld_real_for_choice_%i.png'%which_class)
        plt.savefig(path_fig)
        plt.close()
        return kl_concepts_at_all_times, chosen_classes

def worst_attributes(kl_concepts, nr_experiences, t0, treshold=0.15, which_class=None, perc=0,dir=None):
    data = pd.DataFrame()

    all_kld_errors = np.empty(0)
    j = 1
    for t in range(t0+1, nr_experiences):
        kld_concepts = kl_concepts[j] # / np.max(kld_concepts_0)
        j += 1
        if t == t0 +1:
            sorter = np.argsort(kld_concepts)[::-1]

        x = np.arange(112)[sorter][:10]
        y = kld_concepts[sorter][:10]

        all_kld_errors = np.concatenate( (all_kld_errors, y) )

    l_accepted = len(x)
    data['experience'] = np.sort(np.array( [ a for a in range(t0+1,nr_experiences)]*l_accepted ))
    data['concepts'] = [a for a in x]*(nr_experiences - t0 -1)
    data['kld_values'] = all_kld_errors

    assert len(data['concepts']) == len(data['experience']), str(len(data['concepts']))+' '+str(len(data['experience']))
    assert len(data['concepts']) == len(data['kld_values']), str(len(data['concepts']))+' '+str(len(data['kld_values']))
    ax = sns.barplot(x='concepts', y="kld_values", hue='experience', data=data, saturation=1, errcolor='.3', palette=colors[t0+1:])

    plt.show()
    if which_class is None:
        path_fig = os.path.join(dir, 'Worst_at_%i_kld.png'%t0)
    else:
        path_fig = os.path.join(dir, 'Worst_at_%i_kld_which%i.png'%(t0, which_class))
        plt.title('Class %i'%which_class )
    plt.savefig(path_fig)
    plt.close()
