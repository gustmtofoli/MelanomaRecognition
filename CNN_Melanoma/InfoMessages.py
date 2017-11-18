def infoFolder(path, type):
	# TODO enum
    if type == "TRAIN":
        return print("> [INFO] Training folder:", path)
    elif type == "TEST":
        return print("> [INFO] Testing folder:", path)


def infoLoadImages(imageList, type):
    if type == "TRAIN":
        return print("> [INFO]", len(imageList), "Training images:\n\t" + "[OK] Loaded\n\t" + "[OK] Reshaped")
    elif type == "TEST":
        return print("> [INFO]", len(imageList), "Testing images:\n\t" + "[OK] Loaded\n\t" + "[OK] Reshaped")


def infoImages(imageList):
    return print("> [INFO] Number of images:",len(imageList))


def infoGroundTruth(path_ground_truth, type):
    if type == "TRAIN":
        return print("> [OK] Train Ground Truth File loaded")
    elif type == "TEST":
        return print("> [OK] Train Ground Truth File loaded")


def infoTrainingAccuracy(step, accuracy):
    return print("step %d, training accuracy %g"%(step, accuracy))


def infoTestAccuracy(accuracy):
    return print("test accuracy %g"%(accuracy))

def infoBackgroundRemoved(n_images):
    return print("> [OK] Background removed of ", n_images, " images")


def errorMessage():
    return print("> [ERROR] Still not specified error")

