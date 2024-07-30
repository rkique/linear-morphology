import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
def proj_matrices(matrices, proj_matrix):
    matrices = [np.dot(matrix, proj_matrix.T) for matrix in matrices]
    return matrices
    
def make_basis(x,y):
    typ_x = x / 100 #np.random.rand(4096)
    typ_y = y / 100 #np.random.rand(4096)
    # ymax = np.max(typ_y)
    
    #print(f'{typ_y=}')
    #print(f'{np.linalg.norm(typ_y)=}')

    #normalize basis
    typ_x /= np.linalg.norm(typ_x)
    typ_y /= np.linalg.norm(typ_y)

    pre_orthog = cosine_similarity(typ_x.reshape(1,-1), typ_y.reshape(1,-1))[0,0]
    
    #orthogonalize y with Gram-Schmidt
    typ_y = typ_y - (np.dot(typ_y, typ_x.T) / np.dot(typ_x, typ_x.T)) * typ_x

    #normalize basis
    #This can lead to really bad results when the eigenvectors are really similar...
    typ_x /= np.linalg.norm(typ_x)
    #print(f'dividing {typ_y} by {np.linalg.norm(typ_y)})')
    typ_y /= np.linalg.norm(typ_y)
    
    #print(f'xshape {typ_x.shape} yshape {typ_y.shape}')
    
    post_orthog = cosine_similarity(typ_x.reshape(1,-1), typ_y.reshape(1,-1))[0,0]
    print(f'cosine similarities {pre_orthog=:.3f} {post_orthog=:.3f}')
    return np.array([typ_x, typ_y])
    
def polygon(points):
    centroid = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]
    polygon_points = np.vstack([sorted_points, sorted_points[0]])
    return polygon_points


#Convert cartesian coordinates to polar ones.
def cart2pol(x, y):
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (r, phi)
    
def viz(x_min, x_max, y_min,y_max, title, objs_to_viz,
        proj_subj_hss=None,
        proj_no_bias_hss=None,
        proj_scaled_no_bias_hss=None,
        proj_reg_hss=None,
        proj_obj_hss=None,
        proj_fullobj_hss=None,
        proj_beta_hss_s=None, **kwargs):
    
    annotations = []
    fig = plt.figure(figsize=(6,6))
    # fig, ax = plt.subplots(subplot_kw={"projection": 'polar'})
    
    ##################################
    ###     Polygon Projections    ###
    ##################################
    
    if 'polygon' in kwargs.keys():
        #Plot polygon shapes
        # pg = polygon(proj_no_bias_hss)
        # plt.plot(pg[:, 0], pg[:, 1], 'k-')
        for proj_beta_hss in proj_beta_hss_s:
            pg = polygon(proj_beta_hss)
            plt.plot(pg[:, 0], pg[:, 1], 'k-', linewidth=0.5)
            
        pg = polygon(proj_subj_hss)
        plt.plot(pg[:, 0], pg[:, 1], color='lightcoral')
        
        pg = polygon(proj_no_bias_hss)
        plt.plot(pg[:, 0], pg[:, 1], color='red', linewidth=0.5)
    
        pg = polygon(proj_obj_hss)
        plt.plot(pg[:, 0], pg[:, 1], color='green', linestyle='dashed', linewidth=0.5)

    if 'arrow' in kwargs.keys():
        plt.scatter(0,0, marker='+', alpha=0.8)

    ##################################
    ###            Ws GREY         ###
    ##################################

    x0 = proj_subj_hss[:,0]
    y0 = proj_subj_hss[:,1]
    for text, x, y in zip(objs_to_viz, x0, y0):
        if 'arrow' in kwargs.keys():
            plt.arrow(x,y, -x, -y, color='lightcoral', alpha=0.5)
        annotations.append(plt.annotate(text, 
                     (x, y), 
                     fontsize=6,
                     color='lightgrey', 
                     va='center',
                     alpha=0.8))
        
    ##################################
    ###            Ws PINK         ###
    ##################################
    
    if proj_no_bias_hss is not None and 'ws' in kwargs.keys():
        
        x1 = proj_no_bias_hss[:,0]
        y1 = proj_no_bias_hss[:,1]
        for text, x, y in zip(objs_to_viz, x1, y1):
            annotations.append(plt.annotate(text, 
                         (x,y), 
                         fontsize=6,
                         color='lightcoral', 
                         va='center',
                         alpha=0.8))
    
    ##################################
    ###           BWs PINK        ###
    ##################################
    
    if proj_scaled_no_bias_hss is not None and 'Bws' in kwargs.keys():
        
        x2 = proj_scaled_no_bias_hss[:,0]
        y2 = proj_scaled_no_bias_hss[:,1]
        for text, x, y in zip(objs_to_viz, x2, y2):
            if 'arrow' in kwargs.keys():
                plt.arrow(0,0, x, y, color='lightcoral', alpha=0.5)
            annotations.append(plt.annotate(text, 
                         (x, y), 
                         fontsize=6,
                         color='lightcoral', 
                         va='center',
                         alpha=0.8))
    
    
    ##################################
    ###  Ws + b RED TEXT + ARROWS  ###
    ##################################
    
    if proj_reg_hss is not None and 'reg' in kwargs.keys():
        x2 = proj_reg_hss[:,0]
        y2 = proj_reg_hss[:,1]
        
        for text, x, y in zip(objs_to_viz, x2, y2):
            annotations.append(plt.annotate(text, 
                         (x, y), 
                         fontsize=6,
                         color='red', 
                         va='center',
                         alpha=0.8))
    
    ##################################
    ###      object BLUE TEXT      ###
    ##################################
    
    if proj_obj_hss is not None and 'obj' in kwargs.keys():
        #objects
        x3 = proj_obj_hss[:,0]
        y3 = proj_obj_hss[:,1]
        
        for text, _x, _y, x, y in zip(objs_to_viz, x2,y2, x3, y3):
            # plt.arrow(_x, _y, x-_x, y-_y, color='blue', alpha=0.5)
            annotations.append(plt.annotate(text,
                                (x, y),
                                fontsize=6,
                                color='blue',
                                va='center',
                                alpha=0.8))
    
    ##################################
    ###   object LIGHT BLUE TEXT   ###
    ##################################
    
    if proj_fullobj_hss is not None and 'fullobj' in kwargs.keys():
        x4 = proj_fullobj_hss[:,0]
        y4 = proj_fullobj_hss[:,1]
        print(f'proj full obj is annotating')
        for text, x, y in zip(objs_to_viz, x4, y4):
            plt.arrow(0,0, x * 20, y * 20, color='lightblue', alpha=0.5)
            #plt.arrow(_x, _y, x-_x, y-_y, color='blue', alpha=0.5)
            annotations.append(plt.annotate(text,
                                (x,y),
                                fontsize=6,
                                color='lightblue',
                                va='center',
                                alpha=0.8))
    
    #adjust_text(annotations, arrowprops=dict(arrowstyle="->", color='r', lw=0.5), max_move=(100,100))
    
    ax = plt.gca()
    coords = np.vstack((proj_subj_hss, proj_no_bias_hss, proj_reg_hss))
    
    #center = np.mean(coords, axis=0)
    # ax.set_xlim(center[0] - DIST/2, center[0] + DIST * 2)
    # ax.set_ylim(center[1] - 3, center[1] + 3)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    plt.title(title)
    plt.savefig(f'space_viz/{title}.png')
    plt.show()