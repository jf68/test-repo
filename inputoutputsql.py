
import numpy as np
import pandas as pd

#control_stores = [x for x in control_stores if x not in test_stores]

# Takes a cursor (WD or EDW connector) as input
# and outputs a dataframe	
def exportsql(cursor):
	pd.set_option('display.float_format', lambda x: '%.2f' % x)		
	
	data = cursor.fetchall()
	col_headers = [column[0] for column in cursor.description]

	df = pd.DataFrame(data)
	df.columns = col_headers

	return df
	
# Takes a cursor (WD or EDW connector) and a dataframe as input
# and stores data into a temp table in the correct database
def importsql(cursor, df, temp_table_name):
	
	# Fix data types for columns
	cols = df.columns.values
	
	# Make sure these columns are strings
	order_columns = ['ordno','itemordno','order_num']
	id_columns = ['indid','acctno','hhid','addrid']
	custom_columns = order_columns + id_columns
	
	for c in cols:
		if (c not in custom_columns):
			try:
				df[c] = df[c].astype(float)
			except:
				df[c] = df[c].astype(str)
		else:
			if (df[c].dtypes == 'float64'):
				df[c] = df[c].astype(int).astype(str)
			else:
				df[c] = df[c].astype(str)
				
	sqltypes = []
	for c in cols:
		if (df[c].dtypes == 'float64'):
			sqltypes.append('float')
		else:
			sqltypes.append('varchar('+str(df[c].map(len).max())+')')
			df[c] = "'" + df[c] + "'"
	
	
	# Make the 'create table' statement
	create_statement = "create table " + temp_table_name + " ("
	for i in range(0,len(cols)):
		if i < len(cols)-1:
			create_statement = create_statement + cols[i] + " " + sqltypes[i] + ", "
		else: 
			create_statement = create_statement + cols[i] + " " + sqltypes[i] + ");"		
	#print create_statement
	
	
	# Make the 'insert values into table' statement
	insert_string = "insert into " + temp_table_name + " values ("
	df=df.astype(str)
	df['inserts'] = insert_string + df[cols].apply(lambda x: ', '.join(x), axis=1) + ");"
	insert_statement = df['inserts'].str.cat(sep='\n')
	#print insert_statement	
	
	cursor.execute(create_statement)
	cursor.execute(insert_statement)
	
	
	
	return 0
