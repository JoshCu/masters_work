#open all 20 csv files and combine them into one file
#open the file to write
f = open('combined.csv', 'w')
#write the header
#loop through the 20 files
for i in range(20):
    #open the file to read
    f2 = open('users'+str(i)+'.csv', 'r')
    f2.readline()
    #loop through the lines in the file
    for line in f2:
        #write the line to the combined file
        f.write(line)
    #close the file
    f2.close()
