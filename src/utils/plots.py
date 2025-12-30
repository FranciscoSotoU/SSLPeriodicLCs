import copy
import numpy as np
import matplotlib.pyplot as plt
import time as t

CODE_DICT_INV = {
    0: "LPV",
    1: "RSCVn",
    2: "CEP",
    3: "EA",
    4: "EB/EW",
    5: "RRLc",
    6: "RRLab",
    7: "DSCT",
    8: "Periodic-Other",
}

def plot_aug(lc, aug, aug_name, error=None, fold=False,add_error=False,two_plots=False,code_dict_inv=CODE_DICT_INV):
    label = int(lc['label'])
    # Extract data
    # Split and mask data for both bands
    lc = copy.deepcopy(lc)
    time = lc['time']
    mag = lc['data']
    mask = lc['mask']

    # Process original data
    orig_data = {
        'g': {'time': time[mask==1], 'mag': mag[mask==1]},
        'r': {'time': time[mask==2], 'mag': mag[mask==2]}
    }
    
    if error is not None:
        orig_data['g']['error'] = error[mask==1]
        orig_data['r']['error'] = error[mask==2]
    
    # Process augmented data
    # Calculate elapsed time
    start_time = t.time()
    aug_lc = aug(lc)
    end_time = t.time()
    elapsed_time = end_time - start_time
    print(f"Augmentation time: {elapsed_time:.4f} seconds")
    aug_time = aug_lc['time']
    aug_mag = aug_lc['data']
    aug_mask = aug_lc['mask']
    aug_period = aug_lc['period']


    aug_data = {
        'g': {'time': aug_time[aug_mask==1], 'mag': aug_mag[aug_mask==1]},
        'r': {'time': aug_time[aug_mask==2], 'mag': aug_mag[aug_mask==2]} 
    }
    
    
    if fold:
        for data_dict in [orig_data, aug_data]:
            for band in ['g', 'r']:
                period_val = period if data_dict is orig_data else aug_period
                period_val_div = period_val if aug_name != 'Stretch' else 1
                data_dict[band]['time'] = (data_dict[band]['time'] % period_val) / period_val_div

    # Plotting
    if aug_name == 'Random Noise' and error is not None and not two_plots:
        plt.figure(figsize=(10, 5))
        for band, color, marker in [('g', 'g', 'o'), ('r', 'r', 'o')]:
            if error is not None:
                plt.errorbar(orig_data[band]['time'], orig_data[band]['mag'], 
                           yerr=np.abs(orig_data[band]['error']), 
                           fmt=marker, label=band, color=color, capsize=3, alpha=0.5)
            else:
                plt.scatter(orig_data[band]['time'], orig_data[band]['mag'],
                          marker=marker, label=band, color=color, alpha=0.5)
        for band, color, marker in [('g', 'b', 'x'), ('r', 'black', 'x')]:
            plt.scatter(aug_data[band]['time'], aug_data[band]['mag'], 
                       label=f'{band}_aug', color=color, marker=marker, s=50)
        plt.title(f'Class: {code_dict_inv[label]}\nAugmentation: {aug_name}')
        plt.xlabel('Time')
        plt.ylabel('Magnitude')
        plt.legend(loc='upper right')
        plt.grid(color = 'black', linestyle = '--', linewidth = 0.2)
        #set background color to light grey
        plt.gca().set_facecolor('lightgrey')
        plt.show()
        plt.close()

        
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        for ax, data, title in [(ax1, orig_data, 'Original Data'), 
                               (ax2, aug_data, 'Augmented Data')]:
            for band, color in [('g', 'g'), ('r', 'r')]:
                if error is not None:
                    ax.errorbar(data[band]['time'], data[band]['mag'], 
                              yerr=np.abs(data[band]['error']), 
                              fmt='o', label=band, color=color, capsize=3)
                else:
                    ax.scatter(data[band]['time'], data[band]['mag'],
                             label=band, color=color)
            ax.set_title(title, fontsize=14)
            ax.set_xlabel('Time')
            ax.set_ylabel('Magnitude')
            ax.legend()
        
        if aug_name == 'Time Inverse':
            ax2.invert_xaxis()
        else:
            all_x_lims = np.array([*ax1.get_xlim(), *ax2.get_xlim()])
            all_y_lims = np.array([*ax1.get_ylim(), *ax2.get_ylim()])
            x_lims = (np.min(all_x_lims), np.max(all_x_lims))
            y_lims = (np.min(all_y_lims), np.max(all_y_lims))
            ax1.set_xlim(x_lims)
            ax1.set_ylim(y_lims)
            ax2.set_xlim(x_lims)
            ax2.set_ylim(y_lims)
            
            
        plt.suptitle(f'Class: {code_dict_inv[label]}\nAugmentation: {aug_name}', 
                    y=1.05, fontsize=16)
        #add grid in color blue 
        ax1.grid(color = 'black', linestyle = '--', linewidth = 0.2)
        ax2.grid(color = 'black', linestyle = '--', linewidth = 0.2)

        ax1.set_facecolor('lightgrey')
        ax2.set_facecolor('lightgrey')

        plt.tight_layout()
    plt.show()
    plt.close()



def plot_lc(lc, error=None, period=False, code_dict_inv=CODE_DICT_INV,label=None):
    lc = copy.deepcopy(lc)
    time = lc['time']
    mag = lc['data']
    mask = lc['bands']

    orig_data = {
        'g': {'time': time[mask==1], 'mag': mag[mask==1]},
        'r': {'time': time[mask==2], 'mag': mag[mask==2]}
    }
    
    
    if period:
        for band in ['g', 'r']:
            orig_data[band]['time'] = (orig_data[band]['time'] % period) / period

    plt.figure(figsize=(10, 5))
    for band, color in [('g', 'g'), ('r', 'r')]:
        plt.scatter(orig_data[band]['time'], orig_data[band]['mag'],
                    label=band, color=color)
    
    plt.title(f'Class: {code_dict_inv[label]}')
    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    plt.legend(loc='upper right')
    plt.grid(color='black', linestyle='--', linewidth=0.2)
    plt.gca().set_facecolor('lightgrey')
    plt.show()
    plt.close()


def plot_lc_aug(act_lc, aug_lc, labels_dict, aug_name="Unknown", fold=True):
    """
    Plot the original and augmented light curves side by side with shared x/y ranges.
    """
    # Extract information for title
    period = act_lc.get('features', None)
    if period is not None:
        period = period.numpy()
        if hasattr(period, 'item'):
            period = period.item()
        elif hasattr(period, '__len__') and len(period) > 0:
            period = period[0]

    label = act_lc['label']
    class_name = labels_dict.get(label, f"Class_{label}")
    oid = aug_lc.get('oid', 'Unknown')
    if hasattr(oid, 'decode'):
        oid = oid.decode('utf-8')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    act_lc_copy = copy.deepcopy(act_lc)
    time = act_lc_copy['time']
    mag = act_lc_copy['data']
    mask = act_lc_copy['bands']
    error = act_lc_copy.get('error')

    orig_data = {
        'g': {'time': time[mask==1], 'mag': mag[mask==1]},
        'r': {'time': time[mask==2], 'mag': mag[mask==2]}
    }
    
    if error is not None:
        orig_data['g']['error'] = error[mask==1]
        orig_data['r']['error'] = error[mask==2]

    if period is not None and fold:
        for band in ['g', 'r']:
            orig_data[band]['time'] = (orig_data[band]['time'] % period) / period

    aug_lc_copy = copy.deepcopy(aug_lc)
    aug_time = aug_lc_copy['time']
    aug_mag = aug_lc_copy['data']
    aug_mask = aug_lc_copy['bands']
    aug_error = aug_lc_copy.get('error')

    aug_data = {
        'g': {'time': aug_time[aug_mask==1], 'mag': aug_mag[aug_mask==1]},
        'r': {'time': aug_time[aug_mask==2], 'mag': aug_mag[aug_mask==2]}
    }
    
    if aug_error is not None:
        aug_data['g']['error'] = aug_error[aug_mask==1]
        aug_data['r']['error'] = aug_error[aug_mask==2]

    if period is not None and fold:
        for band in ['g', 'r']:
            aug_data[band]['time'] = (aug_data[band]['time'] % period) / period

    # Gather all x and y values for range calculation
    all_x = np.concatenate([orig_data['g']['time'], orig_data['r']['time'],
                            aug_data['g']['time'], aug_data['r']['time']])
    all_y = np.concatenate([orig_data['g']['mag'], orig_data['r']['mag'],
                            aug_data['g']['mag'], aug_data['r']['mag']])
    x_lim = (np.min(all_x), np.max(all_x))
    y_lim = (np.min(all_y), np.max(all_y))

    for band, color in [('g', 'green'), ('r', 'red')]:
        n_points = len(orig_data[band]['time'])
        if error is not None and 'error' in orig_data[band]:
            ax1.errorbar(orig_data[band]['time'], orig_data[band]['mag'],
                        yerr=np.abs(orig_data[band]['error']),
                        fmt='o', label=f'{band}-band (n={n_points})', 
                        color=color, alpha=0.7, capsize=3, markersize=4)
        else:
            ax1.scatter(orig_data[band]['time'], orig_data[band]['mag'],
                       label=f'{band}-band (n={n_points})', color=color, alpha=0.7, s=30)
    ax1.set_title('Original Light Curve', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Phase' if period is not None else 'Time', fontsize=12)
    ax1.set_ylabel('Magnitude', fontsize=12)
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax1.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    ax1.set_facecolor('#f8f8f8')
    ax1.set_xlim(x_lim)
    ax1.set_ylim(y_lim)

    for band, color in [('g', 'green'), ('r', 'red')]:
        n_points = len(aug_data[band]['time'])
        if aug_error is not None and 'error' in aug_data[band]:
            ax2.errorbar(aug_data[band]['time'], aug_data[band]['mag'],
                        yerr=np.abs(aug_data[band]['error']),
                        fmt='o', label=f'{band}-band (n={n_points})', 
                        color=color, alpha=0.7, capsize=3, markersize=4)
        else:
            ax2.scatter(aug_data[band]['time'], aug_data[band]['mag'],
                       label=f'{band}-band (n={n_points})', color=color, alpha=0.7, s=30)
    ax2.set_title(f'Augmented: {aug_name}', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Phase' if period is not None else 'Time', fontsize=12)
    ax2.set_ylabel('Magnitude', fontsize=12)
    ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax2.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    ax2.set_facecolor('#f8f8f8')
    ax2.set_xlim(x_lim)
    ax2.set_ylim(y_lim)

    period_str = f"Period: {period:.4f}" if period is not None else "Period: N/A"
    total_points_orig = len(act_lc_copy['time'])
    total_points_aug = len(aug_lc_copy['time'])

    plt.suptitle(f'Class: {class_name} | {period_str} | OID: {oid}\n'
                f'Original: {total_points_orig} points | Augmented: {total_points_aug} points', 
                fontsize=14, fontweight='bold', y=1.05)

    plt.tight_layout()
    plt.show()
    plt.close()