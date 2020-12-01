import os
import random

cardboard = os.listdir('dataset/cardboard')
glass = os.listdir('dataset/glass')
metal = os.listdir('dataset/metal')
paper = os.listdir('dataset/paper')
plastic = os.listdir('dataset/plastic')

seven_cardboard = int(len(cardboard))
print('Cardboard: {length:4d} {eight} {twenty}'.format(length=seven_cardboard,
	eight=int(len(cardboard)*0.8),
	twenty=(((len(cardboard))-int(len(cardboard)*0.8)))))

seven_glass = int(len(glass))
print('Glass: {length:8d} {eight} {twenty}'.format(length=seven_glass,
	eight=int(len(glass)*0.8),
	twenty=(((len(glass))-int(len(glass)*0.8)))))

seven_metal = int(len(metal))
print('Metal: {length:8d} {eight} {twenty}'.format(length=seven_metal,
	eight=int(len(metal)*0.8),
	twenty=(((len(metal))-int(len(metal)*0.8)))))

seven_paper = int(len(paper))
print('Paper: {length:8d} {eight} {twenty}'.format(length=seven_paper,
	eight=int(len(paper)*0.8),
	twenty=(((len(paper))-int(len(paper)*0.8)))))

seven_plastic = int(len(plastic))
print('Plastic: {length:6d} {eight} {twenty}'.format(length=seven_plastic,
	eight=int(len(plastic)*0.8),
	twenty=(((len(plastic))-int(len(plastic)*0.8)))))


# CUR_DIRECTORY = os.path.abspath(os.getcwd())

# if(len(os.listdir('dataset/train/cardboard')) != 192):
# 	for _ in range(int(seven_cardboard*0.8)):
# 		choice = random.choice(cardboard)
# 		cardboard.remove(choice)
# 		os.rename(CUR_DIRECTORY+'/dataset/cardboard/'+choice,
# 			CUR_DIRECTORY+'/dataset/train/cardboard/'+choice)
# 	print(len(os.listdir('dataset/train/cardboard')))

# if(len(os.listdir('dataset/train/glass')) != 178):
# 	for _ in range(int(seven_glass*0.8)):
# 		choice = random.choice(glass)
# 		glass.remove(choice)
# 		os.rename(CUR_DIRECTORY+'/dataset/glass/'+choice,
# 		CUR_DIRECTORY+'/dataset/train/glass/'+choice)
# 	print(len(os.listdir('dataset/train/glass')))

# if(len(os.listdir('dataset/train/metal')) != 144):
# 	for _ in range(int(seven_metal*0.8)):
# 		choice = random.choice(metal)
# 		metal.remove(choice)
# 		os.rename(CUR_DIRECTORY+'/dataset/metal/'+choice,
# 		CUR_DIRECTORY+'/dataset/train/metal/'+choice)
# 	print(len(os.listdir('dataset/train/metal')))

# if(len(os.listdir('dataset/train/paper')) != 196):
# 	for _ in range(int(seven_paper*0.8)):
# 		choice = random.choice(paper)
# 		paper.remove(choice)
# 		os.rename(CUR_DIRECTORY+'/dataset/paper/'+choice,
# 		CUR_DIRECTORY+'/dataset/train/paper/'+choice)
# 	print(len(os.listdir('dataset/train/paper')))

# if(len(os.listdir('dataset/train/plastic')) != 340):
# 	for _ in range(int(seven_plastic*0.8)):
# 		choice = random.choice(plastic)
# 		plastic.remove(choice)
# 		os.rename(CUR_DIRECTORY+'/dataset/plastic/'+choice,
# 		CUR_DIRECTORY+'/dataset/train/plastic/'+choice)
# 	print(len(os.listdir('dataset/train/plastic')))

# cardboard = os.listdir('dataset/cardboard')
# glass = os.listdir('dataset/glass')
# metal = os.listdir('dataset/metal')
# paper = os.listdir('dataset/paper')
# plastic = os.listdir('dataset/plastic')

# thirty_cardboard = len(cardboard)
# thirty_glass = len(glass)
# thirty_metal = len(metal)
# thirty_paper = len(paper)
# thirty_plastic = len(plastic)
# print(thirty_cardboard)
# print(thirty_glass)
# print(thirty_metal)
# print(thirty_paper)
# print(thirty_plastic)

# if(thirty_cardboard!=0):
# 	for x in cardboard:
# 		os.rename(CUR_DIRECTORY+'/dataset/cardboard/'+x,
# 		CUR_DIRECTORY+'/dataset/test/cardboard/'+x)
# 	print(len(os.listdir('dataset/test/cardboard')))

# if(thirty_glass!=0):
# 	for x in glass:
# 		os.rename(CUR_DIRECTORY+'/dataset/glass/'+x,
# 		CUR_DIRECTORY+'/dataset/test/glass/'+x)
# 	print(len(os.listdir('dataset/test/glass')))

# if(thirty_metal!=0):
# 	for x in metal:
# 		os.rename(CUR_DIRECTORY+'/dataset/metal/'+x,
# 		CUR_DIRECTORY+'/dataset/test/metal/'+x)
# 	print(len(os.listdir('dataset/test/metal')))

# if(thirty_paper!=0):
# 	for x in paper:
# 		os.rename(CUR_DIRECTORY+'/dataset/paper/'+x,
# 		CUR_DIRECTORY+'/dataset/test/paper/'+x)
# 	print(len(os.listdir('dataset/test/paper')))

# if(thirty_plastic!=0):
# 	for x in plastic:
# 		os.rename(CUR_DIRECTORY+'/dataset/plastic/'+x,
# 		CUR_DIRECTORY+'/dataset/test/plastic/'+x)
# 	print(len(os.listdir('dataset/test/plastic')))

# try:
# 	os.rmdir('dataset/cardboard')
# except OSError as e:
# 	print("Error: %s : %s" % (dir_path, e.strerror))

# try:
# 	os.rmdir('dataset/glass')
# except OSError as e:
# 	print("Error: %s : %s" % (dir_path, e.strerror))

# try:
# 	os.rmdir('dataset/metal')
# except OSError as e:
# 	print("Error: %s : %s" % (dir_path, e.strerror))

# try:
# 	os.rmdir('dataset/paper')
# except OSError as e:
# 	print("Error: %s : %s" % (dir_path, e.strerror))
	
# try:
# 	os.rmdir('dataset/plastic')
# except OSError as e:
# 	print("Error: %s : %s" % (dir_path, e.strerror))