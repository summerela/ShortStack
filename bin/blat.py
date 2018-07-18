import mysql.connector
cnx = mysql.connector.connect(user='nanostring', password = 'user', host='Loki', database='BISProd')
cursor = cnx.cursor()
import os,sys
from BIS_py import percentID,generate_coords

genome = 'hg38'
fname = sys.argv[1]

seqs = {}
f=open(fname,'r')
out = open('to_blat.fsa','w')
for line in f:
    line=line.strip()
    name,seq = line.split('\t')
    seqs[name]=seq
    out.write('>'+name+'\n'+seq+'\n')
f.close()
out.close()

query = "select BLAT_Server, BLAT_Port from CNV_Genome where Build = '"+genome+"'"
cursor.execute(query)
server = ''
port = ''
for a,b in cursor:
    server = a
    port = b

os.system('gfClient '+server+' '+port+' / to_blat.fsa blat_out.txt -out=blast8 -minScore=20')



hits = {}
f = open('blat_out.txt','r')
out = open('blat_result.txt','w')
out.write('Seq Name\tCoords\tStrand\tPercent ID\tHit Number\tMismatches\tGaps\n')
for line in f:
    line = line.strip()
    qid,sid,percid, aln_len,mm,gap,qstart,qend,sstart,send,eval,bit = line.split("\t")
    pid = percentID(len(seqs[qid]),aln_len,percid)
    strand = ''
    coords = ''
    if int(sstart)<int(send):
        coords = generate_coords(sid,sstart,send)
        strand = "+"
    else:
        coords = generate_coords(sid,send,sstart)
        strand ='-'
    if qid not in hits:
        hits[qid] = 0
    hits[qid] = hits[qid]+1 
    out.write(qid+'\t'+coords+'\t'+strand+'\t'+str(pid)+'\t'+str(hits[qid])+'\t'+mm+'\t'+gap+'\n')
f.close()
out.close()
