import csv
import json
import io
import re
from collections import OrderedDict
def s6csvtojson(s6File,jsonfile):
    #Read in s6 CSV
    with open(s6File) as f:
        reader = csv.reader(f)
        Header = next(reader)
        #Get rid of extra comma at end of standard s6 file
        del Header[-1:]
        #Get list of cycle numbers
        CycleList = [int(re.search('C(.*)P', name).group(1)) if "C" in name else name for name in Header ]
        #Get list of pool IDs
        PoolIDList = [int(name.split('P')[1]) if "C" in name else name for name in Header]
        #Sort Cyclelist and remove feature columns to get barcode only data. 
        #This is used to build out dictionary structure so things are ordered correctly. 
        TotalCycles = sorted(set(CycleList[4:]))
        #Establish dictionary that will hold all the data.
        TotalDict = OrderedDict()
        Qual = "999"
        Category = "000"
        FeatureCount = 0

        for row in reader:
            #Get information on FOVID and features, put it into dictionary
            FOVid = row[1]
            X = int(row[2])
            Y = int(row[3])
            FeatureID = str(FOVid) + "_" + str(X) + "_" + str(Y)
            if FOVid not in TotalDict:
                TotalDict[FOVid] = OrderedDict([("FovID",int(FOVid)),("Features",[])])
                FeatureCount = 0
            #Build out structure that will hold all information for this particular feature
            TotalDict[FOVid]['Features'].append(OrderedDict([("FeatureID",FeatureID),("X",X),("Y",Y),("Cycles",[OrderedDict([('CycleID',cycle),("Pools",[])]) for cycle in TotalCycles])]))         

            #Iterate through each column and retrieve barcode information
            for column in range(4,len(Header)-1):
                #Add additional zeros as necessary for cutoff barcodes
                RawBarcode = row[column]
                if len(RawBarcode) == 2:
                    Barcode = "0000" + RawBarcode
                elif len(RawBarcode) == 4:
                    Barcode = "00" + RawBarcode
                else:
                    Barcode = RawBarcode
                #Pretty terrible loop through all the cycles until it matches with the cycle current column, then breaks.
                #PoolIDList index is equivalent to CycleDict index in terms of which column of the s6 they come from
                # Adds barcode information accordingly 
                #. Probably a much better way to do this
                # 
                for CycleDict in TotalDict[FOVid]['Features'][FeatureCount]['Cycles']:
                    if CycleDict.get('CycleID', None) == CycleList[column]:
                        CycleDict['Pools'].append({"PoolID":PoolIDList[column],"BC":Barcode,"Qual":Qual,"Category":Category})
                        break
            FeatureCount += 1

    #Check for unicode? Probably unnecessary, borrowed it from some StackOverflow code
    try:
        to_unicode = unicode
    except NameError:
        to_unicode = str

    # Define data. Defaulting to FOV of 1 here, but you'd probably need to add a loop to generate more jsons for the other
    # FOVs

    data = TotalDict['1']

    # Write JSON file
    with io.open(jsonfile, 'w+', encoding='utf8') as outfile:
        str_ = json.dumps(data,
                          indent=4, sort_keys=False,
                          separators=(',', ': '), ensure_ascii=False)
        outfile.write(to_unicode(str_))

s6csvtojson("/home/selasady/selasady/shortstack_python3/pete/input/temp_s6.csv", "/home/selasady/selasady/shortstack_python3/pete/input/pete_S6.json")
