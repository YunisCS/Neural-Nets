--- RNN1.py	(original)
+++ RNN1.py	(refactored)
@@ -17,7 +17,7 @@
 
 largest_number = pow(2,binary_dim)
 binary = np.unpackbits(
-    np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
+    np.array([list(range(largest_number))],dtype=np.uint8).T,axis=1)
 for i in range(largest_number):
     int2binary[i] = binary[i]
 
@@ -116,13 +116,13 @@
     
     # print out progress
     if(j % 1000 == 0):
-        print "Error:" + str(overallError)
-        print "Pred:" + str(d)
-        print "True:" + str(c)
+        print("Error:" + str(overallError))
+        print("Pred:" + str(d))
+        print("True:" + str(c))
         out = 0
         for index,x in enumerate(reversed(d)):
             out += x*pow(2,index)
-        print str(a_int) + " + " + str(b_int) + " = " + str(out)
-        print "------------"
+        print(str(a_int) + " + " + str(b_int) + " = " + str(out))
+        print("------------")
 
         
