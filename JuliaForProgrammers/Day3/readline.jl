# Julia program to calculate sum of
# 5 integers obtained from console/user input

result = 0

# Prompt to enter
println("Enter 5 numbers line by line")

# Taking Input from user
for number in 1:5
    
    num = readline()
    num = parse(Int64, num)
    global result += num
end

println("The sum is:", result)
