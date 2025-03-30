#k_means implementation


def k_means(img, k, init_centroids, max_iter=10):
   #KMeans loop
  new_centroids = init_centroids
  for i in range(max_iter):
    centroids = new_centroids
    #Assign Centroids
    centroid_assignments = assign_centroid(img, centroids)
    #Calculate new centroids
    new_centroids = update_centroids(img, centroids, centroid_assignments, k)

  #Round for RGB to int values
  new_centroids = new_centroids.astype(int)
  centroid_assignments = assign_centroid(img, new_centroids)
  return new_centroids, centroid_assignments


def assign_centroid(img, centroids):

  #Setup assignment array
  centroid_assignments = np.zeros(img.shape[0] * img.shape[1]*2)
  centroid_assignments = centroid_assignments.reshape((img.shape[0],img.shape[1],2))

  #Iterate over all the pixels
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):

      #Loop over the centroids
      for x in range(len(centroids)):

        #Calculate the distance
        distance = np.sum(np.linalg.norm(img[i, j] - centroids[x]))

        #In the first pass populate the array
        if(x == 0):
          centroid_assignments[i,j] = (x, distance)

        #Compare old distance with current and update if better
        elif(distance < centroid_assignments[i,j][1]):
          centroid_assignments[i,j] = (x, distance)

  return centroid_assignments

def update_centroids(img, centroids, centroids_assignments, k):
  #Setup new array
  new_centroids = np.zeros(k*3)
  new_centroids = new_centroids.reshape((k,3))

  #Loop through centroids
  for k in range(k):
    count = 0

    #Loop through pixels
    for i in range(img.shape[0]):
      for j in range(img.shape[1]):

        #If the pixel is assigned to this centroid add to sum
        if(centroids_assignments[i,j][0] == k):
          new_centroids[k] += img[i,j]
          count += 1

    #Get average R,G,B
    if(count != 0):
      new_centroids[k] = new_centroids[k] / count
    else:
      new_centroids[k] = centroids[k]

  return new_centroids

def init_centroids_rand(img, k):
  #Get pixels for easy calculations

  pixels = img.shape[0] * img.shape[1]

  #Initialize the centroids randomly as long as they are unique
  centroids = np.unique(img.reshape((-1, 3)), axis = 0)
  np.random.shuffle(centroids)
  centroids = centroids[:k]
  return centroids

def init_centroids_rand_spread(img, k):
  centroids = np.random.rand(k,3)
  centroids = centroids * 255
  centroids = centroids.astype(int)
  return centroids


