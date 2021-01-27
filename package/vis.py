import matplotlib.pyplot as plt

def final_graph (a,b, origin, sm_pred, true_test): #graph
    plt.figure(figsize=(40,10))
    plt.plot(origin.loc[a:b], color='tab:red')
    plt.plot(sm_pred['new_smth'].loc[a:b], color='tab:blue')
    plt.plot(true_test['new_smth'][a:b], color = 'tab:green')
    plt.legend(['Signal', 'Predict', 'True'])