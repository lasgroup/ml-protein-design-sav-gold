#Load required packages ----

library(tidyverse)
library(data.table)
library(stringi)
library(Biostrings)
library(fuzzyjoin)
library(readxl)

analyse_NGS <- function(fwd, rev, plate_barcodes){

#This function assumes that the Illumina NGS results are provided as two separate fastq files with forward and reverse reads from paired-end sequencing (fwd and rev)
#In addition, it requires a csv files that provides the mapping of plate barcodes to plate numbers. An example of such a file is provided.

#>>>Load NGS data ----
#Load read 1
NGS_qual_fwd <- readDNAStringSet(fwd, format = "fastq", with.qualities = TRUE)

NGS_data_fwd <- data.table(name = names(NGS_qual_fwd),
                           fwd = as.character(NGS_qual_fwd),
                           qual_fwd = as.character(mcols(NGS_qual_fwd)$qualities)) %>%
  mutate(name = str_remove(name, " 1:N:0:1"))

#Load read 2
NGS_qual_rev <- readDNAStringSet(rev, format = "fastq", with.qualities = TRUE)

NGS_data_rev <- data.table(name = names(NGS_qual_rev),
                           rev = as.character(NGS_qual_rev),
                           qual_rev = as.character(mcols(NGS_qual_rev)$qualities)) %>%
  mutate(name = str_remove(name, " 2:N:0:1"))


#Combine reads in one data frame
NGS_data <- NGS_data_fwd %>%
  inner_join(NGS_data_rev, by = "name")


#>>>Look-up tables for barcode meanings ----
cols = tibble(col = 1:12, barcode3 = c("CATAGG", "TCCGTT", "CATGCA", "TTGTGG", "TTGCCT", "TTGGTC",
                                       "CTCTTG", "GGACAA", "GACTTC", "CCAATC", "CAACGA", "AAGGCT"))

rows = tibble(row = c("A", "B", "C", "D", "E", "F", "G", "H"),
              barcode4 = c("GGAGAA", "CTGGAA", "TCCGAA", "ACAGTG", "GTCTAG", "CTCTTC", "TATCGC", "AGTAGG"))

ext_rev = tibble(ext_rev = 1:6, barcode1 = c("AGGAA", "AGTGG", "ACGTC", "TCAGC", "CTAGC", "GCTTA"))

ext_fwd = tibble(ext_fwd = c("A", "B", "C", "D", "E", "F"), 
                 barcode2 = c("GGTAC", "AACAC", "CGGTT", "GTCAA", "AAGCG", "CCACA"))


#Load combinations of external barcodes that were used for the plates
ext_plates <- fread(plate_barcodes, header = TRUE) %>%
  pivot_longer(cols = -c("ext_fwd"), values_to = "plate_id", names_to = "ext_rev") %>%
  filter(!is.na(plate_id)) %>%
  unite(ext_fwd, ext_rev, col = "barcode_combi", sep = "")



#>>>Filter reads and extract mutations ----
NGS_processed <- NGS_data %>%
  filter(str_detect(fwd, "GTCACACGTAGCATGTGG"), #primer binding site between barcode 1 and 3
         str_detect(rev, "GAGACCTTGTGTCGATGG"), #primer binding site between barcode 2 and 4
         str_detect(fwd, "GGCCTCGGTGGTGCC")) %>% #region between Sav positions 112 and 118
  mutate(const1 = str_locate(fwd, "GTCACACGTAGCATGTGG")[,1],
         barcode1 = str_sub(fwd, start = const1-5, end = const1-1), #external barcode rev primer
         barcode3 = str_sub(fwd, start = const1+18, end = const1+23), #column barcode
         const2 = str_locate(rev, "GAGACCTTGTGTCGATGG")[,1],
         barcode2 = str_sub(rev, start = const2-5, end = const2-1), #external barcode fwd primer
         barcode4 = str_sub(rev, start = const2+18, end = const2+23), #row barcode
         const3 = str_locate(fwd, "GGCCTCGGTGGTGCC")[,1],
         pos110 = as.character(reverseComplement(DNAStringSet(str_sub(fwd, start = const3+21, end = const3+23)))), #codon at position 110
         pos111 = as.character(reverseComplement(DNAStringSet(str_sub(fwd, start = const3+18, end = const3+20)))), #codon at position 111
         pos112 = as.character(reverseComplement(DNAStringSet(str_sub(fwd, start = const3+15, end = const3+17)))), #codon at position 112
         pos118 = as.character(reverseComplement(DNAStringSet(str_sub(fwd, start = const3-3, end = const3-1)))), #codon at position 118
         pos119 = as.character(reverseComplement(DNAStringSet(str_sub(fwd, start = const3-6, end = const3-4)))), #codon at position 119
         pos120 = as.character(reverseComplement(DNAStringSet(str_sub(fwd, start = const3-9, end = const3-7)))), #codon at position 120
         pos121 = as.character(reverseComplement(DNAStringSet(str_sub(fwd, start = const3-12, end = const3-10)))), #codon at position 121
         pos122 = as.character(reverseComplement(DNAStringSet(str_sub(fwd, start = const3-15, end = const3-13)))), #codon at position 122
         aa111 = as.character(translate(DNAStringSet(pos111), no.init.codon = TRUE)), #amino acid at position 111
         aa112 = as.character(translate(DNAStringSet(pos112), no.init.codon = TRUE)), #amino acid at position 112
         aa118 = as.character(translate(DNAStringSet(pos118), no.init.codon = TRUE)), #amino acid at position 118
         aa119 = as.character(translate(DNAStringSet(pos119), no.init.codon = TRUE)), #amino acid at position 119
         aa121 = as.character(translate(DNAStringSet(pos121), no.init.codon = TRUE)), #amino acid at position 121
         mutant = paste(aa111, aa112, aa118, aa119, aa121, sep = ""),
         q111 = str_sub(qual_fwd, start = const3+18, end = const3+20), #sequencing quality at position 111
         q112 = str_sub(qual_fwd, start = const3+15, end = const3+17), #sequencing quality at position 112
         q118 = str_sub(qual_fwd, start = const3-3, end = const3-1), #sequencing quality at position 118
         q119 = str_sub(qual_fwd, start = const3-6, end = const3-4), #sequencing quality at position 119
         q121 = str_sub(qual_fwd, start = const3-12, end = const3-10)) %>%  #sequencing quality at position 121
  filter(pos110 == "CTG", pos120 == "TGG", pos122 == "TCC") %>% #filter for reads that have the expected sequence at positions 110, 120 and 122 (not mutated)
  select(barcode1, barcode2, barcode3, barcode4, mutant, q111, q112, q118, q119, q121) %>%
  unite(q111, q112, q118, q119, q121, col = "qscores", sep = "") %>%
  mutate(qeval = str_replace_all(qscores, pattern = c("\\?" = "T", #recode all symbols representing a Q-score >= 30
                                                      "\\@" = "T",
                                                      "A" = "T",
                                                      "B" = "T",
                                                      "C" = "T",
                                                      "D" = "T",
                                                      "E" = "T",
                                                      "F" = "T",
                                                      "G" = "T", 
                                                      "H" = "T",
                                                      "I" = "T"))) %>%
  filter(qeval == "TTTTTTTTTTTTTTT") %>% #only keep reads with Q-scores of at least 30 at all mutation sites
  left_join(rows, by = "barcode4") %>%
  left_join(cols, by = "barcode3") %>%
  left_join(ext_fwd, by = "barcode2") %>%
  left_join(ext_rev, by = "barcode1") %>%
  unite(ext_fwd, ext_rev, col = "barcode_combi", remove = FALSE, sep = "") %>%
  left_join(ext_plates, by = "barcode_combi") %>%
  na.omit() %>%
  unite(plate_id, row, col, col = "origin", sep = "_", remove = FALSE) %>%
  mutate(origin = factor(origin))


#>>>Determine correct mutant for each well ----
variants <- NGS_processed %>%
  group_by(origin, mutant) %>%
  summarise(n = n(),
            plate_id = dplyr::first(plate_id),
            ext_fwd = dplyr::first(ext_fwd),
            ext_rev = dplyr::first(ext_rev),
            row = dplyr::first(row),
            col = dplyr::first(col)) %>%
  ungroup() %>%
  filter(n > 2) %>% #minimum number of reads per variant (adjust based on read coverage)
  mutate(plate_mutant = paste(plate_id, mutant, sep = "_")) %>%
  group_by(plate_id, mutant) %>%
  slice_max(order_by = n, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  group_by(origin) %>%
  mutate(n_origin = sum(n),
         percent = n/n_origin*100,
         well = paste(row, col, sep = ""),
         plate_id = factor(plate_id)) %>%
  filter(percent > 80, #Discard data for wells that contain more than one variant
         !str_detect(mutant, pattern = "\\*")) %>% #Remove variants with stop codons
  ungroup()

}


