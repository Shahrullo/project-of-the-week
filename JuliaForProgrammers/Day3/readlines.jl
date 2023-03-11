# Julia program to take
# multi-lined input from user

line_count = 0

println("Enter multi-lined text, press Ctrl-D when done")

# Calling readlines() function
lines = readlines()

# Loop to count lines
for line in lines
	global line_count += 1

end

println("total no.of.lines: ", line_count)

println(lines)

# Getting type of Input values
println("type of input: ", typeof(lines))
