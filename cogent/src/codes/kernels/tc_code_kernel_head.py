#
#
#
def tc_gen_code_Kernel_Head(f, kernel_name, l_t3_d_decl_var, l_t2_d_decl_var, l_v2_d_decl_var, l_input_strides, l_external_idx, l_internal_idx, l_inputs_addr,
                            opt_internal, opt_pre_computed, opt_data_type):
    #
    f.write("\n")
    f.write("// created by tc_gen_code_Kernel()\n")
    f.write("__global__ void ")
    f.write(kernel_name)
    f.write("(")

    # parameters for t3 (output)
    for t3_var in l_t3_d_decl_var:
        if opt_pre_computed == -1:
            if "range"  in t3_var:
                continue
            if "base"   in t3_var:
                continue
            if "offset" in t3_var:
                continue

            #
            f.write(t3_var)
            f.write(", ")
        else:
            f.write(t3_var)
            f.write(", ")
    f.write("\n")

    # parameters for t2 (left)
    for t2_var in l_t2_d_decl_var:
        if opt_pre_computed == -1:
            if "addr"   in t2_var:
                continue
            if "offset" in t2_var:
                continue
            #
            f.write(t2_var)
            f.write(", ")
        else:
            f.write(t2_var)
            f.write(", ")
    f.write("\n")

    # parameters for v2 (right)
    for v2_var in l_v2_d_decl_var:
        if opt_pre_computed == -1:
            if "addr"   in v2_var:
                continue
            if "offset" in v2_var:
                continue
            #
            f.write(v2_var)
            f.write(", ")
        else:
            f.write(v2_var)
            f.write(", ")
    f.write("\n")   

    #
    #   [Rules] Sizes:      External Indinces and Iternal Indices
    #           numBlks:    Externl Indices
    #
    if opt_pre_computed == -1:
        #
        #   [Sizes] External
        #
        for each_idx in l_external_idx:
            f.write("int size_" + each_idx + ", ")

        #
        #   [Sizes] Internal
        #
        for each_idx in l_internal_idx:
            f.write("int size_" + each_idx + ", ")
        f.write("\n")

        #
        #   [Blks] External
        # 
        for each_idx in l_external_idx:
            f.write("int numBlk_" + each_idx + ", ")
        f.write("\n")

    #
    #   (Optional) Strides for LEFT and RIGHT (if internal index is not FVI)
    #   This is for an internal index, because constant-memory will be used for
    #   multiple-internal index.
    #
    if len(l_input_strides) > 0:
        for each_tc in l_input_strides:
            f.write("int " + each_tc[0] + ", ")
            f.write("int " + each_tc[2] + ", ")
        f.write("\n")

    #
    #   (Optional)
    #
    if opt_internal > 1:
        for each_tensor_contraction in l_inputs_addr:
            f.write("int* dev_internal_offset_" + each_tensor_contraction[0][3] + ", ")
            f.write("int* dev_internal_offset_" + each_tensor_contraction[1][3] + ", ")
            f.write("\n")

    #   
    f.write("int stride_reg_x, ")
    f.write("int stride_reg_y, ")
    f.write("\n")

    # the size of |internal indices| = size_internal
    f.write("int size_internal")
    f.write(")\n")

#
#
#
def tc_gen_code_Kernel_Head_RT(f, kernel_name, l_combined_t3_d_decl_var, l_combined_t2_d_decl_var, l_combined_v2_d_decl_var):
    #
    f.write("\n")
    f.write("// created by tc_gen_code_kernel_RT()\n")
    f.write("__global__ void ")
    f.write(kernel_name)
    f.write("(")

    #   T3 (Output)
    #
    idx_count = 0
    for each_inner_group in l_combined_t3_d_decl_var:
        for t3_var in each_inner_group:
            if idx_count != 0:
                idx_count = idx_count - 1   # Need to Get Rid of the Output (Overlapped)
                continue
            f.write(t3_var)
            f.write(", ")
        f.write("\n")
        idx_count = idx_count + 1

    #   T2 (LEFT)
    for each_inner_group in l_combined_t2_d_decl_var:
        for t2_var in each_inner_group:
            f.write(t2_var)
            f.write(", ")
        f.write("\n")

    #   V2 (RIGHT)
    for each_inner_group in l_combined_v2_d_decl_var:
        for v2_var in each_inner_group:
            f.write(v2_var)
            f.write(", ")
        f.write("\n")

    #
    f.write("int stride_reg_x, ")
    f.write("int stride_reg_y, ")

    #
    f.write("int size_internal")
    f.write(")\n")

def tc_gen_code_Kokkos_Kernel_Head(f, kernel_name, l_t3_d_decl_var, l_t2_d_decl_var, l_v2_d_decl_var, l_input_strides, l_external_idx, l_internal_idx, l_inputs_addr,
                            opt_internal, opt_pre_computed, opt_data_type):
    f.write("\n")
    f.write("// created by tc_gen_code_Kokkos_Kernel()\n")
    f.write("struct ")
    f.write(kernel_name)
    f.write("{")
    f.write("\n")
    
    # # Definition of Kokkos View
    f.write("\n")
    f.write("\tKokkos::View<DataType*> dev_t3; \n ")
    f.write("\tKokkos::View<DataType*> dev_t2; \n ")
    f.write("\tKokkos::View<DataType*> dev_v2; \n ")

    
    
    
    
    
    
    #
    #   [Rules] Sizes:      External Indinces and Iternal Indices
    #           numBlks:    Externl Indices
    #
    if opt_pre_computed == -1:
        #
        #   [Sizes] External
        #
        
        f.write("\n")
        f.write("\t// External Indices \n")
        
        for each_idx in l_external_idx:
            f.write("\tint size_" + each_idx + " ; \n")



        #
        #   [Sizes] Internal
        #
        
        f.write("\n")
        f.write("\t// Internal Indices \n")
        
        for each_idx in l_internal_idx:
            f.write("\tint size_" + each_idx + " ; \n")
        # f.write("\n")

        #
        #   [Blks] External
        # 
        
        f.write("\n")
        f.write("\t// [Blks] External Indices \n")
        for each_idx in l_external_idx:
            f.write("\tint numBlk_" + each_idx + "; \n ")
        # f.write("\n")

    #
    #   (Optional) Strides for LEFT and RIGHT (if internal index is not FVI)
    #   This is for an internal index, because constant-memory will be used for
    #   multiple-internal index.
    #
    if len(l_input_strides) > 0:
        for each_tc in l_input_strides:
            f.write("\tint " + each_tc[0] + "; \n ")
            f.write("\tint " + each_tc[2] + "; \n ")
        # f.write("\n")

    #
    #   (Optional)
    # 
    f.write("\n")
    if opt_internal > 1:
        for each_tensor_contraction in l_inputs_addr:
            f.write("\tKokkos::View<int*> dev_internal_offset_" + each_tensor_contraction[0][3] + "; \n ")
            f.write("\tKokkos::View<int*> dev_internal_offset_" + each_tensor_contraction[1][3] + "; \n ")
            
            # f.write("\n")

    #   
    f.write("\n")
    f.write("\tint stride_reg_x ; \n ")
    f.write("\tint stride_reg_y ; \n ")
    # f.write("\n")

    # the size of |internal indices| = size_internal
    f.write("\tint size_internal ; \n")
    
    f.write("\n")
    
    # Creat Constructor
    f.write("\t"+kernel_name+"( DualViewVectorType& dev_32_ , DualViewVectorType& dev_v2_ , DualViewVectorType& dev_t2_ , \n")
    
    for idx, each_idx in enumerate(l_external_idx):
        if idx == 0:
            f.write("\t\tint size_" + each_idx + "_ ,")
        else:
            f.write("int size_" + each_idx + "_ ,")
            
            
    f.write("\n")        
    for idx, each_idx in enumerate(l_internal_idx):
        if idx == 0:
            f.write("\t\tint size_" + each_idx + "_ ,")
        else:
            f.write("int size_" + each_idx + "_ ,")
            
    f.write("\n")    
    for idx, each_idx in enumerate(l_external_idx):
        if idx == 0:
            f.write("\t\tint numBlk_" + each_idx + "_ ,")
        else:
            f.write("int numBlk_" + each_idx + "_ ,")
            
    f.write("\n")     
    if opt_internal > 1:
        for each_tensor_contraction in l_inputs_addr:
            f.write("\t\tDualViewVectorTypeInt& host_internal_offset_" + each_tensor_contraction[0][3] + ", \n ")
            f.write("\t\tDualViewVectorTypeInt&  host_internal_offset_" + each_tensor_contraction[1][3] + ", \n ")
            
    
    # f.write("\n")      
    f.write("\t\tint stride_reg_x_ , ")
    f.write(" int stride_reg_y_ , \n ")
    f.write("\t\tsize_internal_ )  \n ")
    
    f.write("\t\t\t:   dev_t3(dev_t3_.d_view) , dev_t2(dev_t2_.d_view) , dev_v2(dev_v2_.d_view) , \n")
    
    for idx, each_idx in enumerate(l_external_idx):
        if idx == 0:
            f.write("\t\t\tsize_" + each_idx + "(size_" + each_idx + "_) , ")
        else:
            f.write("size_" + each_idx + "(size_" + each_idx + "_) , ")
            
    f.write("\n")        
    for idx, each_idx in enumerate(l_internal_idx):
        if idx == 0:
            f.write("\t\t\tsize_" + each_idx + "(size_" + each_idx + "_) , ")
        else:
            f.write("size_" + each_idx + "(size_" + each_idx + "_) , ")
            
    f.write("\n")    
    for idx, each_idx in enumerate(l_external_idx):
        if idx == 0:
            f.write("\t\t\tnumBlk_" + each_idx + "(size_" + each_idx + "_) , ")
        else:
            f.write("numBlk_" + each_idx + "(size_" + each_idx + "_) , ")

             
    # f.write("\n")     
    # if opt_internal > 1:
    #     for tensor_contraction in enumerate(l_inputs_addr):
    #         if idx == 0:
    #             f.write("\t\tdev_internal_offset_" + tensor_contraction[0][3] + "(host_internal_offset_"+ tensor_contraction[0][3] + ".d_view" + ", ")
    #         else:
    #             f.write("\t\tdev_internal_offset_" + tensor_contraction[1][3] + "(host_internal_offset_"+ tensor_contraction[1][3] + ".d_view" + ", ")
    f.write("\n")    
    if opt_internal > 1:  
        f.write("\t\t\tdev_internal_offset_t2(host_internal_offset_t2.d_view) , ")      
        f.write("dev_internal_offset_v2(host_internal_offset_v2.d_view) , \n ")        
        
    f.write("\t\t\tstride_reg_x(stride_reg_x_) ,")
    f.write("stride_reg_y(stride_reg_y) , \n ")
    
    f.write("\t\t\tsize_internal(size_internal_) \n")
    f.write("\t\t\t{\n")
    f.write("\t\t\t\tdev_t3_.modify<Device>(); \n")
    f.write("\t\t\t}\n")
    
    