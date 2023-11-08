'''
'''

#region: single_chemical_pod_data
def single_chemical_pod_data(ax, chem_pod_data, color='#B22222'):
    '''
    '''
    ax.scatter(
        chem_pod_data['POD'], 
        chem_pod_data['Cum_Proportion'], 
        color=color, 
        zorder=5  # zorder ensures it's on top
    )
#endregion

#region: single_chemical_moe_data
def single_chemical_moe_data(ax, chem_moe_data, color='#B22222'):
    '''
    '''
    # Plot the central tendencies as  markers
    ax.scatter(
        chem_moe_data['POD_50%ile'], 
        chem_moe_data['Cum_Count'], 
        color=color, 
        zorder=5  # zorder ensures it's on top
    )

    # Plot horizontal lines representing the POD prediction interval
    # For each percentile of exposure uncertainty
    for cum_count, lb, ub in zip(
            chem_moe_data['Cum_Count'], 
            chem_moe_data['POD_5%ile'], 
            chem_moe_data['POD_95%ile']
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