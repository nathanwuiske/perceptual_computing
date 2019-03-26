
import numpy as np
import cv2

test_image = ''
standard_deck_image = ''
number_of_cards = 54
cards_to_search = 4

def labeledCards(standard_deck_image, number_of_cards):
    '''First input: image of the entire deck of cards
       Second input: number of cards within the deck. default is 54'''
    labels = labelCards()
    labeled_cards = {}
    image = cv2.imread(standard_deck_image)
    '''Extract all the cards within the deck and iterate over them '''
    for i, indiv_card in enumerate(convert_cards(image, number_of_cards)): 
        '''Create a new dictionary of labelled cards. Each key is associated with a label and the preproccessed card.
           preprocess returns a thresholded image'''
        labeled_cards[i] = (labels[i], preprocess(indiv_card))
    return (labeled_cards)

def featureMatch(standard_deck_image, number_of_cards, input_card):
    '''Third input: input test image'''
    labels = labelCards()
    '''construct a SIFT and BFMatcher object. Create a dictionary to store all the best matches'''
    sift = cv2.xfeatures2d.SIFT_create()
    matcher = cv2.BFMatcher()
    best = {}
    image = cv2.imread(standard_deck_image)
    '''Extract all the cards within the deck and iterate over them '''
    for i, indiv_card in enumerate(convert_cards(image, number_of_cards)): 
        bestMatches = []
        '''find descriptors'''
        keypoints1, descriptor1 = sift.detectAndCompute(input_card, None)
        keypoints2, descriptor2 = sift.detectAndCompute(indiv_card, None)
        '''take the descriptor of one feature in first card and match with all other features in second card. Uses
           cv2.NORM_L2 (euclidean distance) as the distance calculation
           return k best matches'''
        matches = matcher.knnMatch(descriptor1, descriptor2, k=2)
        ''' checks if matches are ambiguous and should be removed.
            Compare the distance of the closest neighbor to that of the second closest neighbor
            reject all matches in which the ratio is greater than 0.8
          '''
        for k,j in matches:
            if k.distance < 0.8 * j.distance:  # distance between descriptors. lower = better
                bestMatches.append([k])
        #img3 = cv2.drawMatchesKnn(input_card,keypoints1,indiv_card,keypoints2,matches,None,flags=2)
        #cv2.imshow('2', img3)
        #cv2.waitKey(0) 
        '''add to the best matches dictionary, taking the length of bestMatches '''
        best[i] = (labels[i], len(bestMatches))
    '''Sort by bestMatches length. Higher length = more elements = more likely to be the test input image card '''
    sorted_Best = sorted(best.items(), key=lambda s: s[1][1], reverse=True)[:1] 
    '''return the card label associated with the highest length '''
    return (sorted_Best[0][1]) # return first tuple (the label) 

def preprocess(image):
    blur_image = cv2.GaussianBlur(image, (5,5), 2) # 5x5 kernel, standard deviation in X and Y direction = 2
    ret_thresh = cv2.adaptiveThreshold(blur_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)
    return ret_thresh
  
def order_points(points): 
    '''Create a list of co-ordinates called 'points', that will be ordered the following way:
       first entry = top left, second entry = top right, third entry = bottom right, fourth entry = bottom left'''
    points = points.reshape((4,2)) 
    new = np.zeros((4,2),dtype = np.float32) 
    add = points.sum(1) 
    new[0] = points[np.argmin(add)] # top left has the smallest sum
    new[2] = points[np.argmax(add)] # bottom right has the largest sum
    diff = np.diff(points, axis = 1) 
    new[1] = points[np.argmin(diff)] # top right has the smallest diff
    new[3] = points[np.argmax(diff)] # bottom left has the largest diff
    return new

def convert_cards(image, numcards):
    '''First input: image to extract cards from  
       Second input: number cards within the image
       Perform basic preprocessing and find all contours in the image. This gives the edges to the cards themselves
       and the numbers/symbols within them. This returns a list of all contours in the image where each indivdual contour
       is a list of coordinates that represent the boundary points. Sorting by contour area will give the largest area. This
       will be the card within the image.
       '''
    imgSize = 449
    imCopy = image.copy()
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (1,1), 5)
    ret, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)  

    image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    #img3 = cv2.drawContours(imCopy, contours, -1, (0,255,0), 3)
    #cv2.imshow('2', thresh)
    #cv2.waitKey(0) 
    contours = sorted(contours, key=cv2.contourArea,reverse=True)[:numcards]  
    for card in contours:
        '''For each card found, create a rectangular representation of it.
           A contour approximation can be made with a specific precision, epsilon. This was set to 1% of the perimeter
           cv2.approxPolyDP returns four points specifying the (x, y) coordinates of each point of the rectangle.
           The ordering needs to be consistent for both the dst and approx, they both use the following:
           Element: 1 = top left, 2 = top right, 3 = bottom right, 4 = bottom left
           '''
        epsilon = 0.01*cv2.arcLength(card,True)
        approx = order_points(cv2.approxPolyDP(card, epsilon, True))
        approx1 = cv2.approxPolyDP(card, epsilon, True)
        
        # transformation 
        '''4 points representing the top-down view of the image. Uses the same ordering as before
          Element: 1 = top left, 2 = top right, 3 = bottom right, 4 = bottom left
        '''
        dst = np.array([[0,0],[imgSize,0],[imgSize,imgSize],[0,imgSize]], np.float32)  
        '''Get the transformation matrix'''
        transform = cv2.getPerspectiveTransform(approx, dst)
        '''the warped image is found using the transformation matrix and image, returned as a 450x450 img'''
        warp = cv2.warpPerspective(image,transform,(450,450))
        yield warp  


def gaussian_blur(image, kernel_size, sigma):
     return cv2.GaussianBlur(image, (kernel_size,kernel_size), sigma)
    
def abs_diff(image1, image2):
    '''Two inputs: two images to be compared
       Blur both images and calculate the per-element absolute difference between the two images
       blur this difference again to further remove noise/artifacts then perform basic thresholding
       Return summed image threshold, which is sorted in the next function to find minimal difference. 
       '''
    image1 = gaussian_blur(image1, 5, 4)
    image2 = gaussian_blur(image2, 5, 4)
    diff = cv2.absdiff(image1, image2)  
    blur = gaussian_blur(diff, 5, 4)
    ret, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY) 
    return np.sum(thresh)  

def nearest_card(labeled_cards, image):
    ''' First input: a dictionary of all the labeled_cards returned from labeledCards function
        Second input: an image of the card(s) to be recognised from the test input image
        -Preprocess the test input image, returns a threshold image
        -Use the second (indexed at 1) value of the dictionary which is the threshold images 
        -Compare the two using the absolute difference function. Repeat with all 54 cards
         and return the first element which is the minimal difference.
        '''
    test_card = preprocess(image)
    return sorted(labeled_cards.values(), key=lambda s: abs_diff(s[1], test_card))[0][0]

def labelCards():
    ''' When inputting the standard_deck_image into the convert_cards function, the cards come out in an arbitrary way.
        The labels list contains all the cards in the order that they were proccessed. '''
    labels = {}
    labels[0] = ('A', 'H')
    labels[1] = ('2', 'H')
    labels[2] = ('3', 'H')
    labels[3] = ('A', 'D')
    labels[4] = ('4', 'H')
    labels[5] = ('2', 'D')
    labels[6] = ('5', 'H')
    labels[7] = ('3', 'D')
    labels[8] = ('Q', 'H')
    labels[9] = ('K', 'H')
    labels[10] = ('6', 'H')
    labels[11] = ('A', 'C')
    labels[12] = ('J', 'H')
    labels[13] = ('JKR', 'JKR')
    labels[14] = ('10', 'H')
    labels[15] = ('7', 'H')
    labels[16] = ('9', 'H')
    labels[17] = ('2', 'C')
    labels[18] = ('8', 'H')
    labels[19] = ('4', 'D')
    labels[20] = ('3', 'C')
    labels[21] = ('5', 'D')
    labels[22] = ('K', 'D')
    labels[23] = ('A', 'S')
    labels[24] = ('JKR', 'JKR')
    labels[25] = ('Q', 'D')
    labels[26] = ('6', 'D')
    labels[27] = ('J', 'D')
    labels[28] = ('2', 'S')
    labels[29] = ('7', 'D')
    labels[30] = ('4', 'C')
    labels[31] = ('10', 'D')
    labels[32] = ('8', 'D')
    labels[33] = ('9', 'D')
    labels[34] = ('3', 'S')
    labels[35] = ('5', 'C')
    labels[36] = ('K', 'C')
    labels[37] = ('Q', 'C')
    labels[38] = ('4', 'S')
    labels[39] = ('6', 'C')
    labels[40] = ('J', 'C')
    labels[41] = ('7', 'C')
    labels[42] = ('10', 'C')
    labels[43] = ('8', 'C')
    labels[44] = ('9', 'C')
    labels[45] = ('5', 'S')
    labels[46] = ('K', 'S')
    labels[47] = ('6', 'S')
    labels[48] = ('Q', 'S')
    labels[49] = ('J', 'S')
    labels[50] = ('7', 'S')
    labels[51] = ('8', 'S')
    labels[52] = ('9', 'S')
    labels[53] = ('10', 'S')
    return labels

'''Flip the image if the image width is greater than the image height.
   result is an image rotated 90 degrees''' 
image = cv2.imread(test_image) 
w = image.shape[1]
h = image.shape[0]
if w > h:
    image = cv2.transpose(image) 
    image = cv2.flip(image, 1) # 1 = vertically
    
'''Start brute force feature matching method''' 
brute_force = [featureMatch(standard_deck_image, number_of_cards, f) for f in convert_cards(image, cards_to_search)]
print('Brute Force:')
print('\n'.join('Card {}: {}'.format(*i) for i in enumerate(brute_force)))

'''Start absolute difference method''' 
labeled_cards = labeledCards(standard_deck_image, number_of_cards)
print('Absolute Difference:')
absolute_diff = [nearest_card(labeled_cards, f) for f in convert_cards(image, cards_to_search)]
print('\n'.join('Card {}: {} '.format(*i) for i in enumerate(absolute_diff)))

