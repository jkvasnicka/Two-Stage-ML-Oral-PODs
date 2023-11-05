'''
'''

#region: single_chemical_pod_data
def single_chemical_pod_data(ax, chem_pod_data, color='#B22222'):
    '''
    '''
    ax.scatter(
        chem_pod_data['pod'], 
        chem_pod_data['cum_proportion'], 
        color=color, 
        zorder=5  # zorder ensures it's on top
    )
#endregion

# TODO: Swap the level at data storage
#region: single_chemical_moe_data
def single_chemical_moe_data(ax, chem_moe_data, color='#B22222'):
    '''
    '''
    chem_moe_data = chem_moe_data.swaplevel()  # for convenience

    # Plot the central tendencies as  markers
    ax.scatter(
        chem_moe_data['moe'], 
        chem_moe_data['cum_count'], 
        color=color, 
        zorder=5  # zorder ensures it's on top
    )

    # Plot horizontal lines representing the POD prediction interval
    for cum_count, lb, ub in zip(
        chem_moe_data['cum_count'], chem_moe_data['lb'], chem_moe_data['ub']
        ):
        ax.hlines(
            cum_count, 
            lb, 
            ub, 
            colors=color, 
            lw=1, 
            zorder=4
        )
#endregion