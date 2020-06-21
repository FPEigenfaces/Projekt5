from sklearn.decomposition import PCA
import numpy as np

def compute_eigenfaces(train_faces, img_size, num_eigenfaces):
    faceVecLength = img_size[0]*img_size[1]
    returnMatrix = np.zeros((len(train_faces), faceVecLength))
    for i, faceVec in enumerate(train_faces):
        flattenedVec = faceVec.flatten()
        returnMatrix[i] = flattenedVec

    avg_face = returnMatrix.mean(axis = 0)
    returnMatrix = returnMatrix - avg_face

    pca = PCA(n_components=num_eigenfaces, svd_solver='randomized', whiten=True).fit(returnMatrix)
    eigenfaces = pca.components_.reshape((num_eigenfaces, img_size[0], img_size[1]))
    return avg_face, eigenfaces

def reconstruct_img(img, avg_face, e_faces, img_size):
    projection = get_projection(img, avg_face, e_faces)
    reconstruction = e_faces.T.dot(projection)
    reconstruction = reconstruction + avg_face
    reconstruction = reconstruction.reshape(img_size)
    
    return reconstruction

def get_projection(img, avg_face, e_faces):
    mean_free_img_vec = (img.flatten() - avg_face)
    proj_weigths = np.dot(e_faces, mean_free_img_vec)
    return proj_weigths

def classify_face(img, reconstructed_img, threshold=1000):
    diff = img - reconstructed_img
    diff_sum = diff.sum()
    if diff_sum <= threshold:
        return (True, diff_sum)
    else:
        return (False, diff_sum)
   
def get_train_projection(train_images, avg_face, e_faces):        
    train_proj = []
    
    for train_img in train_images:
        train_proj.append(get_projection(train_img, avg_face, e_faces))
    
    train_proj_weights = np.array(train_proj)
    return train_proj_weights
    
def get_most_similar_face_id(img, avg_face, e_faces, train_proj_weights):
    projected_img = get_projection(img, avg_face, e_faces)
    distance = ssd.cdist(np.atleast_2d(projected_img), train_proj_weights, metric = 'cityblock')
    return distance.argmin()