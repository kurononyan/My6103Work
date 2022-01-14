#%%
print("Hello world")
print(5 / 8)
print (7+10)
print(10/3, 3/10) # print numerical divisions
print(10//3, 3//10) # print quotients from divisions
print(10%3, 3%10) # print remainders from divisions

#%%
astring = "Thank you"
anum = 3.14159265358979323846
cnt = 1
# Many different ways to print out the same line
print("%d. I want to say %s" % (cnt,astring) )
cnt+=1
print(cnt,". I want to say" + astring )
cnt+=1
print(cnt, ". I want to say",astring )
cnt+=1
print("%d. I want to say %s, my sweetie %.3f" % (cnt,astring,anum) )
cnt+=1
print("%d. I want to say %s, my sweetie digit %d" % (cnt,astring,anum) )
cnt+=1
print("%d. I want to say %s, my sweetie long %f" % (cnt,astring,anum) )
cnt+=1
# For python 3.6+, we can use the f-string
print(f"{cnt}. I want to say {astring}, my sweetie long {anum.__round__(3)}")
cnt+=1
# see https://python-reference.readthedocs.io/en/latest/docs/str/formatting.html
# s-string, d-digit (int), f-float
# 
# side note:  
# for more info on the new python f-string since python 3.6, see
# for example: https://cito.github.io/blog/f-strings/
#
# also, see https://docs.python.org/3/reference/lexical_analysis.html#literals
# for info on f-string, r-string, b-string, etc.


#%%[markdown]
#
# # Python Class 01
# 
# This is our first class of the semester.
# Hello to everyone.   
# Two spaces in the previous line doesn't make a new line in this environment. 
#
# You will need a blank line to get a new paragraph.

# The above is not considered a blank line without the # sign.
#
# This can get you a [link](http://www.gwu.edu).
#
# You can find some cheatsheets to do other basic stuff like bold-face, italicize, tables, etc.

# %%
