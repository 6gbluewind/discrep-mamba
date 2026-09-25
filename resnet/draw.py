import seaborn as sns
import matplotlib.pyplot as plt

def plot_rsa_matrix(matrix, labels, save_path, title=None, cmap="RdBu_r"):

    plt.figure(figsize=(8, 8))
    sns.set_theme(style="white")
    
    ax = sns.heatmap(
        matrix, 
        annot=True, 
        fmt=".3f", 
        cmap=cmap,
        center=0,
        vmin=-1, vmax=1,
        xticklabels=labels, 
        yticklabels=labels,
        square=True,
        linewidths=1.5,  
        annot_kws={
            "size": 14,
            "weight": "bold" 
        },
        cbar_kws={"shrink": .75}
    )

    cbar = ax.collections[0].colorbar
    cbar.set_label('Pearson Correlation (r)', 
                   size=14, 
                   fontweight='bold', 
                   labelpad=15)
    
    cbar.ax.tick_params(labelsize=12)
    for t in cbar.ax.get_yticklabels():
        t.set_fontweight('bold')

    plt.xticks(fontsize=12, fontweight='bold', rotation=0)
    plt.yticks(fontsize=12, fontweight='bold', rotation=0)

    # if title:
    #     plt.title(title, fontsize=15, fontweight='bold', pad=25)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"RSA Matrix saved to: {save_path}")


 